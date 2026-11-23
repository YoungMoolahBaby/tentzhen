"""The LittleBit layer engine (docs/littlebit.md, "Hardware architecture"): one LittleBit layer
from `integer.json`, its integers hardwired into logic (Variant A), applied to a stream of input
vectors. An MLP layer is the one-vector case; a convolution streams the window around each
position of its grid (docs/littlebit.md, "After MNIST").

It computes what the golden model computes (`LittleBitLayer::trace` in
crates/littlebit/src/integer.rs), in three stages a vector:

1. V, one input per clock as it arrives, `inputs` clocks, and one more: p = x * g_i, and each of
   the `rank` accumulators a_k adds p, or subtracts it, by V's row i. The input and its rows of g
   and V are kept for a clock first, so each add comes a clock after its input.
2. ℓ, one latent per clock, `rank` clocks: t_k = (a_k * l_k) >> latent_shift, saturated to an
   int16 and kept, with its negation.
3. U and h, one output per clock, `outputs` clocks: U's row o is read; a clock later s_o, the sum
   of the `rank` latents, each added or subtracted by that row, in one adder tree; a clock later
   z_o = s_o * h_o + bias_o, and y_o = z_o >> output_shift, clamped to 0..255. Each output comes
   out with `y_valid`, the last with `y_last` too.

Stage 3 sums each output's latents at once, rather than keeping an accumulator for every output
and adding one latent to all of them each clock in stage 2. Synthesized for r16's first layer,
the 256 accumulators that would take came to 18,778 LUT4s and 3,106 carry cells (Yosys, OSS CAD
Suite 2026-11-21), past what both layers may use (docs/littlebit.md, "Hardware architecture").
The clocks kept between steps are where nextpnr's first routes of the two layers fell short of
50 MHz: 40.72 and 44.80 MHz.

Every right shift is arithmetic, as the golden model's are. The sums are as wide as they can get,
so none wraps. The a's shift along in stage 2, so it reads only the first, and the latents shift
in behind them. The engine takes no input in stages 2 and 3. Given an input every clock, a vector
takes inputs + rank + outputs + 1 clocks, its last output comes two clocks after that, and the
next vector's first input is taken on the first of those two.

The ROMs, `ROM`, are trees of multiplexers on constants, so Yosys makes them logic. That the
engine matches the golden model, and takes those clocks, is `tested` on its test vectors
(tests/test_littlebit.py, from crates/littlebit/src/vectors.rs). What it costs in LUTs, and how
fast it closes, are nextpnr's to report: `python -m gateware.cores.littlebit` writes an engine
alone as Verilog, for Yosys and nextpnr.
"""

from amaranth import Const, Elaboratable, Module, Mux, Signal, signed, unsigned

__all__ = ["ROM", "LittleBitEngine", "Alone", "sign_bits"]

# The largest product of a uint8 input and an int8 scale, in size: 255 * 128.
P_MAX = 255 * 128
T_MIN, T_MAX = -(1 << 15), (1 << 15) - 1


def sign_bits(signs):
    """Signs as integer.json writes them, a string of `+` and `-`, as an integer whose bit k is 1
    where sign k is `-`."""
    if set(signs) - {"+", "-"}:
        raise ValueError(f"a sign is + or -, not one of {sorted(set(signs) - {'+', '-'})}")
    return sum(1 << k for k, c in enumerate(signs) if c == "-")


class ROM(Elaboratable):
    """Row `index` of `rows`, integers of shape `shape`, on `data`, as a tree of 2:1 multiplexers
    on the constant rows, one level for each bit of the index: the rows become logic (Variant A).
    Past the last row, `data` is 0.

    Not a Switch with a constant on each Case: Yosys's proc_rom pass turns such a Switch into a
    memory, and once the memory's output is registered, synth_ecp5 put it in block RAM. Four of
    layer 1's ROMs went into 4 DP16KD blocks that way (OSS CAD Suite 2026-11-21), which would
    make it Variant B. proc_rom works on a process's switches only (passes/proc/proc_rom.cc), and
    a tree of multiplexers is none."""

    def __init__(self, rows, shape):
        self.rows = list(rows)
        self.index = Signal(range(max(len(self.rows), 2)))
        self.data = Signal(shape)

    def elaborate(self, platform):
        m = Module()
        shape = self.data.shape()
        level = [Const(row, shape) for row in self.rows]
        for bit in range(len(self.index)):
            pairs = zip(level[0::2], level[1::2] + [Const(0, shape)] * (len(level) % 2))
            level = [Mux(self.index[bit], high, low) for low, high in pairs]
        m.d.comb += self.data.eq(level[0])
        return m


def signed_sum(terms):
    """The sum of `terms`, as a balanced tree of adds."""
    while len(terms) > 1:
        terms = [a + b for a, b in zip(terms[0::2], terms[1::2])] + terms[len(terms) & ~1 :]
    return terms[0]


class LittleBitEngine(Elaboratable):
    """One LittleBit layer, `layer` as integer.json holds it (a dict), applied to each vector of
    uint8s streamed into `x`.

    In: `x`, taken on a clock when `x_valid` and `x_ready` are both high, a vector's inputs in
    order. Out: `y`, a vector's outputs in order, each on a clock `y_valid` is high, the last with
    `y_last`; and beside each, `z`, the int32 before the shift and clamp, for tests to compare."""

    def __init__(self, layer):
        self.inputs = layer["inputs"]
        self.outputs = layer["outputs"]
        self.rank = layer["rank"]
        if len(layer["g"]) != self.inputs or len(layer["v"]) != self.inputs:
            raise ValueError("g or V is not as long as the layer's inputs")
        if len(layer["l"]) != self.rank:
            raise ValueError("l is not as long as the layer's rank")
        if any(len(x) != self.outputs for x in (layer["u"], layer["h"], layer["bias"])):
            raise ValueError("U, h or the bias is not as long as the layer's outputs")
        if any(len(row) != self.rank for row in layer["v"] + layer["u"]):
            raise ValueError("a row of V or U is not as long as the layer's rank")
        self.layer = layer
        self.x = Signal(8)
        self.x_valid = Signal()
        self.x_ready = Signal()
        self.y = Signal(8)
        self.z = Signal(signed(32))
        self.y_valid = Signal()
        self.y_last = Signal()

    def ports(self):
        return [self.x, self.x_valid, self.x_ready, self.y, self.z, self.y_valid, self.y_last]

    def elaborate(self, platform):
        m = Module()
        layer, n_in, n_out, rank = self.layer, self.inputs, self.outputs, self.rank

        # V's and U's rows, one per input and one per output: bit k is 1 where sign k is `-`.
        m.submodules.g = g = ROM(layer["g"], signed(8))
        m.submodules.v = v = ROM([sign_bits(row) for row in layer["v"]], unsigned(rank))
        m.submodules.l = l = ROM(layer["l"], signed(8))
        m.submodules.u = u = ROM([sign_bits(row) for row in layer["u"]], unsigned(rank))
        m.submodules.h = h = ROM(layer["h"], signed(8))
        m.submodules.bias = bias = ROM(layer["bias"], signed(32))

        i = Signal(range(max(n_in, 2)))
        k = Signal(range(max(rank, 2)))
        o = Signal(range(max(n_out, 2)))

        a_max = n_in * P_MAX
        s_max = rank * (1 << 15)
        a = [Signal(range(-a_max, a_max + 1), name=f"a_{j}") for j in range(rank)]
        # Each latent, and its negation, kept for stage 3.
        t = [Signal(signed(16), name=f"t_{j}") for j in range(rank)]
        nt = [Signal(signed(17), name=f"nt_{j}") for j in range(rank)]

        # Stage 1, its first clock: the input and its rows of g and V, kept. Its second: the
        # product, a uint8 times an int8, added or subtracted.
        x_1 = Signal(8)
        g_1 = Signal(signed(8))
        v_1 = Signal(unsigned(rank))
        first_1 = Signal()
        valid_1 = Signal()
        p = Signal(signed(17))
        m.d.comb += [g.index.eq(i), v.index.eq(i), p.eq(x_1 * g_1)]
        m.d.sync += valid_1.eq(0)
        with m.If(valid_1):
            for j in range(rank):
                start = Mux(first_1, 0, a[j])
                m.d.sync += a[j].eq(start + Mux(v_1[j], -p, p))

        # Stage 2's latent, from the first a: the product, shifted, saturated to an int16.
        m.d.comb += l.index.eq(k)
        latent = (a[0] * l.data) >> layer["latent_shift"]
        shifted = Signal(latent.shape())
        t_next = Signal(signed(16))
        m.d.comb += shifted.eq(latent)
        with m.If(shifted > T_MAX):
            m.d.comb += t_next.eq(T_MAX)
        with m.Elif(shifted < T_MIN):
            m.d.comb += t_next.eq(T_MIN)
        with m.Else():
            m.d.comb += t_next.eq(shifted)

        # Stage 3, its first clock: output o's row of U, kept. Its second: the sum of the latents
        # by that row, with output o's h and bias. Its third: z exactly, then shifted and clamped.
        u_1 = Signal(unsigned(rank))
        o_1 = Signal.like(o)
        o_2 = Signal.like(o)
        valid_u1 = Signal()
        valid_u2 = Signal()
        s = Signal(range(-s_max, s_max + 1))
        h_2 = Signal(signed(8))
        bias_2 = Signal(signed(32))
        m.d.comb += [u.index.eq(o), h.index.eq(o_1), bias.index.eq(o_1)]
        m.d.sync += [
            valid_u1.eq(0),
            s.eq(signed_sum([Mux(u_1[j], nt[j], t[j]) for j in range(rank)])),
            h_2.eq(h.data),
            bias_2.eq(bias.data),
            o_2.eq(o_1),
            valid_u2.eq(valid_u1),
        ]
        exact = s * h_2 + bias_2
        z_exact = Signal(exact.shape())
        y_shifted = Signal((z_exact >> layer["output_shift"]).shape())
        y_next = Signal(8)
        m.d.comb += [z_exact.eq(exact), y_shifted.eq(z_exact >> layer["output_shift"])]
        with m.If(y_shifted < 0):
            m.d.comb += y_next.eq(0)
        with m.Elif(y_shifted > 255):
            m.d.comb += y_next.eq(255)
        with m.Else():
            m.d.comb += y_next.eq(y_shifted)
        m.d.sync += [
            self.y.eq(y_next),
            self.z.eq(z_exact),
            self.y_valid.eq(valid_u2),
            self.y_last.eq(valid_u2 & (o_2 == n_out - 1)),
        ]

        with m.FSM():
            with m.State("V"):
                m.d.comb += self.x_ready.eq(1)
                with m.If(self.x_valid):
                    m.d.sync += [
                        x_1.eq(self.x),
                        g_1.eq(g.data),
                        v_1.eq(v.data),
                        first_1.eq(i == 0),
                        valid_1.eq(1),
                    ]
                    with m.If(i == n_in - 1):
                        m.d.sync += i.eq(0)
                        m.next = "V-last"
                    with m.Else():
                        m.d.sync += i.eq(i + 1)
            # The clock on which stage 1 adds its last input in.
            with m.State("V-last"):
                m.next = "L"
            with m.State("L"):
                for j in range(rank - 1):
                    m.d.sync += [a[j].eq(a[j + 1]), t[j].eq(t[j + 1]), nt[j].eq(nt[j + 1])]
                m.d.sync += [t[rank - 1].eq(t_next), nt[rank - 1].eq(-t_next)]
                with m.If(k == rank - 1):
                    m.d.sync += k.eq(0)
                    m.next = "U"
                with m.Else():
                    m.d.sync += k.eq(k + 1)
            with m.State("U"):
                m.d.sync += [u_1.eq(u.data), o_1.eq(o), valid_u1.eq(1)]
                with m.If(o == n_out - 1):
                    m.d.sync += o.eq(0)
                    m.next = "V"
                with m.Else():
                    m.d.sync += o.eq(o + 1)
        return m


class Alone(Elaboratable):
    """An engine as its own top, for Yosys and nextpnr to size: its ports, less `z`, which only
    tests read, so that synthesis drops what makes it."""

    def __init__(self, layer):
        self.engine = LittleBitEngine(layer)
        self.x = Signal(8)
        self.x_valid = Signal()
        self.x_ready = Signal()
        self.y = Signal(8)
        self.y_valid = Signal()
        self.y_last = Signal()

    def ports(self):
        return [self.x, self.x_valid, self.x_ready, self.y, self.y_valid, self.y_last]

    def elaborate(self, platform):
        m = Module()
        m.submodules.engine = e = self.engine
        m.d.comb += [
            e.x.eq(self.x),
            e.x_valid.eq(self.x_valid),
            self.x_ready.eq(e.x_ready),
            self.y.eq(e.y),
            self.y_valid.eq(e.y_valid),
            self.y_last.eq(e.y_last),
        ]
        return m


if __name__ == "__main__":
    # python -m gateware.cores.littlebit <integer.json> <layer index> <out.v>, from the repo root.
    # The engine alone, with nothing at its ports: then, for instance,
    #   yosys -p "read_verilog <out.v>; synth_ecp5 -top engine -json <out.json>"
    #   nextpnr-ecp5 --45k --package CABGA381 --speed 6 --freq 50 --seed 1 --json <out.json>
    import json
    import sys
    from pathlib import Path

    from amaranth.back import verilog

    network, index, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    top = Alone(json.loads(Path(network).read_text())["layers"][index])
    out.write_text(verilog.convert(top, name="engine", ports=top.ports()))
