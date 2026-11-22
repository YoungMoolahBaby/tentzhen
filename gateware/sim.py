"""Running a cocotb test module against Amaranth gateware on Verilator, from a pytest test: the
gateware becomes Verilog, Verilator builds it, and cocotb runs the module's tests on it. Under
pytest, cocotb's runner fails the pytest test if any cocotb test in the module fails."""

from amaranth.back import verilog
from cocotb_tools.runner import get_runner


def run(top, ports, name, test_module, build_dir, platform=None):
    """Builds `top`, with `ports` as its toplevel module `name`'s ports, in `build_dir`, and runs
    the cocotb tests in `test_module` on it."""
    source = build_dir / f"{name}.v"
    source.write_text(verilog.convert(top, name=name, ports=ports, platform=platform))
    runner = get_runner("verilator")
    runner.build(
        sources=[source],
        hdl_toplevel=name,
        build_dir=build_dir,
        build_args=["-Wno-fatal"],
        timescale=("1ns", "1ps"),
        always=True,
    )
    runner.test(test_module=test_module, hdl_toplevel=name, build_dir=build_dir)
