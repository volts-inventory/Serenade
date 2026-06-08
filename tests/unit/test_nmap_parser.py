from __future__ import annotations

from tools.nmap_ import _parse_nmap_xml

_SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<nmaprun start="0">
  <host>
    <status state="up"/>
    <address addr="10.10.10.5" addrtype="ipv4"/>
    <hostnames><hostname name="demo.lab"/></hostnames>
    <ports>
      <port protocol="tcp" portid="22">
        <state state="open"/>
        <service name="ssh" product="OpenSSH" version="8.4p1"/>
      </port>
      <port protocol="tcp" portid="80">
        <state state="open"/>
        <service name="http" product="Apache httpd" version="2.4.49"/>
      </port>
      <port protocol="tcp" portid="3306">
        <state state="closed"/>
        <service name="mysql"/>
      </port>
    </ports>
    <os>
      <osmatch name="Linux 5.x"/>
    </os>
  </host>
</nmaprun>
"""


def test_parse_basic_host():
    hosts = _parse_nmap_xml(_SAMPLE_XML)
    assert len(hosts) == 1
    h = hosts[0]
    assert h.ip == "10.10.10.5"
    assert h.state == "up"
    assert "demo.lab" in h.hostnames
    assert h.os_family == "linux"
    assert len(h.ports) == 3
    open_ports = [p for p in h.ports if p.state == "open"]
    assert {p.port for p in open_ports} == {22, 80}
    apache = next(p for p in open_ports if p.port == 80)
    assert apache.product == "Apache httpd"
    assert apache.version == "2.4.49"


def test_parse_empty_returns_empty():
    assert _parse_nmap_xml("") == []
    assert _parse_nmap_xml("not xml") == []


def test_parse_skips_host_without_ip():
    xml = """<?xml version="1.0"?><nmaprun><host><status state="up"/></host></nmaprun>"""
    assert _parse_nmap_xml(xml) == []
