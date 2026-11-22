"""Running cocotb tests against Amaranth gateware on Verilator, from a pytest test: the gateware
becomes Verilog, Verilator builds it, and cocotb runs the named tests on it. Under pytest, cocotb's
runner fails the pytest test if a cocotb test fails, but passes one in which no test ran at all, so
`run` also counts them."""

from amaranth.back import verilog
from cocotb_tools.check_results import get_results
from cocotb_tools.runner import get_runner


def run(top, ports, name, test_module, tests, build_dir, platform=None):
    """Builds `top`, with `ports` as its toplevel module `name`'s ports, in `build_dir`, runs the
    cocotb tests named in `tests` from `test_module` on it, and fails unless each of them ran and
    passed."""
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
    results = runner.test(
        test_module=test_module, testcase=tests, hdl_toplevel=name, build_dir=build_dir
    )
    ran, failed = get_results(results)
    assert (ran, failed) == (len(tests), 0), (
        f"{test_module}: {ran} of the {len(tests)} cocotb tests named ran, and {failed} failed"
    )
