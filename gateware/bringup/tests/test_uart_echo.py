"""A cocotb test of gateware/bringup/uart_echo.py, the board design as it is, on Verilator, run by
pytest. A stand-in platform hands the design plain signals for its UART and its LED. From the repo
root, in OSS CAD Suite's environment:

    python3 -m pytest gateware
"""

import random
from types import SimpleNamespace

import cocotb
from amaranth import Signal
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge

from gateware.bringup.blinky import CLOCK_HZ
from gateware.bringup.uart_echo import BAUD, UARTEcho
from gateware.sim import run

DIVISOR = round(CLOCK_HZ / BAUD)


def test_uart_echo(tmp_path):
    uart_rx, uart_tx, led = Signal(name="uart_rx"), Signal(name="uart_tx"), Signal(name="led")
    pins = {
        "uart": SimpleNamespace(rx=SimpleNamespace(i=uart_rx), tx=SimpleNamespace(o=uart_tx)),
        "led": SimpleNamespace(o=led),
    }
    platform = SimpleNamespace(request=lambda name, number: pins[name])
    tests = ["uart_echo_sends_back_every_byte_at_115200_baud"]
    ports = [uart_rx, uart_tx, led]
    run(UARTEcho(), ports, "uart_echo", "test_uart_echo", tests, tmp_path, platform)


async def cycles(dut, n):
    for _ in range(n):
        await FallingEdge(dut.clk)


@cocotb.test()
async def uart_echo_sends_back_every_byte_at_115200_baud(dut):
    """32 bytes, 0x00, 0xFF, 0x55 and 0xAA among them, sent back to back into the design at 25 MHz
    / 217, the divisor the board runs: each comes back once and in order, read from the design's TX
    line in the middle of each bit."""
    assert DIVISOR == 217
    rng = random.Random(0)
    sent = [0x00, 0xFF, 0x55, 0xAA] + [rng.randrange(256) for _ in range(28)]
    dut.uart_rx.value = 1
    dut.rst.value = 1
    Clock(dut.clk, 40, unit="ns").start()
    await cycles(dut, 3)
    dut.rst.value = 0
    await cycles(dut, 3)

    async def host_sends():
        for byte in sent:
            for bit in [0, *((byte >> i) & 1 for i in range(8)), 1]:
                dut.uart_rx.value = bit
                await cycles(dut, DIVISOR)

    async def host_receives():
        got = []
        for _ in range((len(sent) + 2) * 10 * DIVISOR):
            if len(got) == len(sent):
                break
            await FallingEdge(dut.clk)
            if int(dut.uart_tx.value):
                continue
            await cycles(dut, DIVISOR // 2)
            assert int(dut.uart_tx.value) == 0, "a start bit shorter than half a bit"
            byte = 0
            for i in range(8):
                await cycles(dut, DIVISOR)
                byte |= int(dut.uart_tx.value) << i
            await cycles(dut, DIVISOR)
            assert int(dut.uart_tx.value) == 1, f"no stop bit after {byte:#04x}"
            got.append(byte)
        return got

    cocotb.start_soon(host_sends())
    assert await cocotb.start_soon(host_receives()) == sent
