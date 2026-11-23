"""The PLL test, ahead of the Phase 2 layer engine (docs/littlebit.md, "The board"): Gate 0's echo
and blinky, clocked at 50 MHz from the ECP5's PLL instead of at 25 MHz from the oscillator.

The PLL's dividers are what `ecppll -i 25 -o 50` gave (gateware/cores/ecp5_pll.py): the 25 MHz
oscillator, divided by 1 and multiplied by 2, from a VCO at 600 MHz that CLKOP divides by 12. The
echo's divisor is then round(50 MHz / 115,200) = 434, and the blinky counts 25,000,000 cycles a
half second. Run from 25 MHz instead, the same design would blink once every two seconds and
echo at 57,600 baud, not 115,200. On i9-2 it echoes at 115,200 and not at 57,600, and blinks about
once a second: the PLL locks and the design runs near 50 MHz (measured:
lab/records/2026-11-23-i9-pll-echo.toml). How near is not measured.

This builds the bitstream and nothing more. Loading it is a flash, which a person approves in the
session and the lab log records (lab/targets.toml), done by hand with openFPGALoader. From the
repo root:

    python -m gateware.bringup.pll_echo <build-dir>
"""

import sys

from amaranth import ClockDomain, Elaboratable, Module

from gateware.boards.colorlight_i9 import ColorlightI9Platform
from gateware.bringup.uart_echo import UARTEcho
from gateware.cores.ecp5_pll import PLL

CLOCK_HZ = 50_000_000


def clock_50_mhz(m, platform):
    """Gives module `m` the `sync` domain at 50 MHz, from the PLL fed by the board's 25 MHz
    oscillator, as this design runs it."""
    m.domains.sync = ClockDomain("sync")
    clk25 = platform.request(platform.default_clk, 0)
    f_in = int(platform.default_clk_frequency)
    m.submodules.pll = pll = PLL(
        clk25.i, f_in, clki_div=1, clkfb_div=2, clkop_div=12, clkop_cphase=5
    )
    assert pll.f_out_hz == CLOCK_HZ, f"the PLL gives {pll.f_out_hz} Hz, not {CLOCK_HZ}"


class PLLEcho(Elaboratable):
    """The echo and the blinky, in a 50 MHz domain made by the PLL from the 25 MHz oscillator."""

    def elaborate(self, platform):
        m = Module()
        clock_50_mhz(m, platform)
        m.submodules.echo = UARTEcho(CLOCK_HZ)
        return m


if __name__ == "__main__":
    build_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
    ColorlightI9Platform().build(PLLEcho(), name="pll_echo", build_dir=build_dir)
