"""Phrasing templates. `{entity}` and `{value}` are filled from a planted fact;
multi-fact templates also take `{entity2}` and `{value2}`.

Statement templates imitate how a user states a fact in chat; tool templates
imitate command output and config files a tool returns; claim templates
imitate how an assistant restates a fact (filled with a wrong value, so a
recognized claim should fire drift).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str
    kind: str
    text: str
    role: str = "user"


def _t(kind: str, role: str, pairs: list[tuple[str, str]]) -> list[Template]:
    return [Template(id=i, kind=kind, text=text, role=role) for i, text in pairs]


STATEMENTS: dict[str, list[Template]] = {
    "port": _t(
        "port",
        "user",
        [
            ("running_on", "{entity} is running on port {value}."),
            ("session_listens", "For this session: {entity} listens on port {value}."),
            ("the_port_is", "The {entity} port is {value}."),
            ("colon_label", "{entity} port: {value}"),
            ("anchor_after", "port {value} is where {entity} listens"),
            ("arrow", "{entity} → port {value}"),
            ("bare_colon", "we bound {entity} to :{value}"),
            ("slash_tcp", "{entity} is on {value}/tcp"),
            ("localhost_colon", "Use {entity} at localhost:{value}."),
            ("parenthetical", "{entity} (port {value}) is healthy."),
            ("cli_flag", "start {entity} with --port {value}"),
            ("port_number", "{entity} is exposed on port number {value}"),
            ("previous_sentence", "{entity} is up. It listens on port {value}."),
            ("tcp_port", "The server for {entity} answers on TCP port {value}."),
            ("mid_sentence", "I run {entity} on port {value} and nothing else there."),
            ("backticks", "`{entity}` runs on port `{value}`."),
            ("possessive", "{entity}'s port is {value}"),
        ],
    ),
    "ipv4": _t(
        "ipv4",
        "user",
        [
            ("is_at", "{entity} is at {value}."),
            ("host_has_ip", "The {entity} host has IP {value}."),
            ("colon_label", "{entity} ip: {value}"),
            ("lives_on", "{entity} lives on {value} on the LAN."),
            ("reach_via", "reach {entity} via {value}"),
            ("anchor_after", "{value} is the address of {entity}"),
            ("arrow", "{entity} → {value}"),
            ("ip_address_for", "The IP address for {entity} is {value}."),
            ("backtick_equals", "`{entity}` = {value}"),
            ("previous_sentence", "{entity} is up. Its address is {value}."),
        ],
    ),
    "ipv6": _t(
        "ipv6",
        "user",
        [
            ("is_at", "{entity} is at {value}."),
            ("colon_label", "{entity} ipv6: {value}"),
            ("address_for", "The IPv6 address for {entity} is {value}."),
            ("bracketed", "reach {entity} via [{value}]"),
            ("arrow", "{entity} → {value}"),
            ("anchor_after", "{value} is the address of {entity}"),
        ],
    ),
    "version": _t(
        "version",
        "user",
        [
            ("is_version", "{entity} is version {value}."),
            ("v_prefix", "we run {entity} v{value}"),
            ("name_space_version", "{entity} {value} is installed."),
            ("colon_label", "{entity} version: {value}"),
            ("the_version_is", "The {entity} version is {value}."),
            ("bare_after_verb", "upgraded {entity} to {value}"),
            ("parenthetical", "{entity} (v{value})"),
            ("pinned_at", "{entity} is pinned at {value}."),
            ("backticks", "`{entity}` version {value}"),
            ("previous_sentence", "{entity} is up. It is version {value}."),
            ("pip_pin", "Running {entity}=={value}"),
            ("npm_at", "{entity}@{value}"),
        ],
    ),
    "env_var": _t(
        "env_var",
        "user",
        [
            ("assign", "{entity}={value}"),
            ("set_in_env", "set {entity}={value} in the env"),
            ("is_set_to", "{entity} is set to {value}"),
            ("export", "export {entity}={value}"),
            ("yaml_colon", "{entity}: {value}"),
            ("backticks", "`{entity}={value}`"),
            ("env_var_equals", "the env var {entity} equals {value}"),
            ("spaced_equals", "{entity} = {value}"),
            ("quoted", '{entity}="{value}"'),
        ],
    ),
    "numeric_cfg": _t(
        "numeric_cfg",
        "user",
        [
            ("yaml_colon", "{entity}: {value}"),
            ("spaced_equals", "{entity} = {value}"),
            ("assign", "{entity}={value}"),
            ("set_to", "set {entity} to {value}"),
            ("setting_is", "the {entity} setting is {value}"),
            ("quoted", '{entity}: "{value}"'),
            ("cli_flag", "--{entity} {value}"),
            ("is", "{entity} is {value}"),
            ("backticks", "`{entity}: {value}`"),
        ],
    ),
    "path": _t(
        "path",
        "user",
        [
            ("config_is", "the config is {value}"),
            ("reads", "{entity} reads {value}"),
            ("colon_label", "config path: {value}"),
            ("backticks", "edit `{value}`"),
            ("sentence_end", "see {value}."),
            ("parenthetical", "({value})"),
            ("equals", "file={value}"),
            ("lives_at", "{entity} config lives at {value} on the host"),
        ],
    ),
    "url": _t(
        "url",
        "user",
        [
            ("is_at", "{entity} is at {value}"),
            ("colon_label", "base url: {value}"),
            ("curl", "curl {value}"),
            ("angle_brackets", "use <{value}> for {entity}"),
            ("endpoint_is", "the {entity} endpoint is {value}."),
        ],
    ),
    "hostname": _t(
        "hostname",
        "user",
        [
            ("runs_on", "{entity} runs on {value}"),
            ("colon_label", "host: {value}"),
            ("ssh", "ssh to {value}"),
            ("the_box", "{entity} lives on the box {value}."),
            ("backticks", "`{value}` hosts {entity}"),
        ],
    ),
    "container": _t(
        "container",
        "user",
        [
            ("the_container", "the container {value} runs {entity}"),
            ("is_the_container", "{value} is the {entity} container"),
            ("restart", "restart {value}"),
            ("colon_label", "container: {value}"),
            ("docker_logs", "docker logs {value}"),
            ("backticks", "`{value}`"),
        ],
    ),
}

# Two facts of the same kind in one message: the anchor decides whether a claim
# about the first can still be checked.
MULTI: dict[str, list[Template]] = {
    "port": _t(
        "port",
        "user",
        [
            ("and", "{entity} is on port {value} and {entity2} is on port {value2}."),
            ("list", "ports: {entity} {value}, {entity2} {value2}"),
            ("two_sentences", "{entity} listens on port {value}. {entity2} listens on port {value2}."),
            ("host_colon", "{entity}:{value} and {entity2}:{value2}"),
        ],
    ),
    "version": _t(
        "version",
        "user",
        [
            ("with", "{entity} {value} with {entity2} {value2}"),
            ("two_sentences", "{entity} is version {value}. {entity2} is version {value2}."),
        ],
    ),
    "ipv4": _t(
        "ipv4",
        "user",
        [
            ("and", "{entity} is at {value} and {entity2} is at {value2}."),
        ],
    ),
}

# Tool results: command output and configuration files. `{entity}` is the
# service name, or for env_var / numeric_cfg the variable or key itself.
TOOL: list[Template] = [
    Template("json_nested", "port", '{{"{entity}": {{"host": "0.0.0.0", "port": {value}}}}}', "tool"),
    Template("json_flat", "port", '{{"port": {value}, "service": "{entity}"}}', "tool"),
    Template("yaml_nested", "port", "{entity}:\n  port: {value}\n  threads: 4\n", "tool"),
    Template("yaml_flat", "port", "name: {entity}\nport: {value}\n", "tool"),
    Template("toml_section", "port", "[{entity}]\nport = {value}\n", "tool"),
    Template("compose_ports", "port", 'services:\n  {entity}:\n    ports:\n      - "{value}:{value}"\n', "tool"),
    Template(
        "docker_ps",
        "port",
        "CONTAINER ID   IMAGE             STATUS         PORTS                    NAMES\n"
        "3f9a1c2d4e5b   {entity}:latest   Up 3 hours     0.0.0.0:{value}->{value}/tcp   {entity}\n",
        "tool",
    ),
    Template(
        "ss_tlnp",
        "port",
        'LISTEN 0      4096         0.0.0.0:{value}       0.0.0.0:*    users:(("{entity}",pid=4242,fd=3))\n',
        "tool",
    ),
    Template(
        "systemctl_status",
        "port",
        "● {entity}.service - {entity}\n     Active: active (running) since Fri 2026-09-11 08:00:01 UTC\n"
        "     CGroup: /system.slice/{entity}.service\n             └─4242 /usr/bin/{entity} --port {value} --model /models/x.gguf\n",
        "tool",
    ),
    Template("nginx_listen", "port", "server {{\n    listen {value};\n    server_name {entity};\n}}\n", "tool"),
    Template("nginx_proxy", "port", "location / {{\n    proxy_pass http://127.0.0.1:{value};\n}}\n", "tool"),
    Template("pip_show", "version", "Name: {entity}\nVersion: {value}\nSummary: a package\n", "tool"),
    Template("pip_list", "version", "Package         Version\n--------------- -------\n{entity}        {value}\n", "tool"),
    Template("dpkg_l", "version", "ii  {entity}    {value}    amd64    a package\n", "tool"),
    Template("version_flag", "version", "$ {entity} --version\n{entity} {value}\n", "tool"),
    Template("cargo_toml", "version", '[package]\nname = "{entity}"\nversion = "{value}"\n', "tool"),
    Template("package_json", "version", '{{"name": "{entity}", "version": "{value}"}}', "tool"),
    Template(
        "ip_addr",
        "ipv4",
        "2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500\n    inet {value}/24 brd 10.0.0.255 scope global eth0\n",
        "tool",
    ),
    Template("hosts_file", "ipv4", "127.0.0.1   localhost\n{value}   {entity}\n", "tool"),
    Template("ping", "ipv4", "PING {entity} ({value}) 56(84) bytes of data.\n", "tool"),
    Template("docker_inspect", "ipv4", '            "IPAddress": "{value}",\n', "tool"),
    Template("yaml_numeric", "numeric_cfg", "{entity}: {value}\n", "tool"),
    Template("json_numeric", "numeric_cfg", '{{"{entity}": {value}}}', "tool"),
    Template("toml_numeric", "numeric_cfg", "{entity} = {value}\n", "tool"),
    Template("cli_numeric", "numeric_cfg", "/usr/bin/server --{entity} {value}\n", "tool"),
    Template("env_file", "env_var", "# service\n{entity}={value}\n", "tool"),
    Template("env_yaml", "env_var", 'environment:\n  {entity}: "{value}"\n', "tool"),
    Template("env_compose_list", "env_var", "environment:\n  - {entity}={value}\n", "tool"),
    Template("ls_l", "path", "-rw-r--r-- 1 root root 1234 Sep 12 10:00 {value}\n", "tool"),
    Template("find_output", "path", "{value}\n", "tool"),
    Template("head_header", "path", "==> {value} <==\n", "tool"),
]

# How an assistant mentions a *different* value without contradicting the
# fact: suggestions, defaults, hypotheticals, examples. Filled with a wrong
# value; drift firing on any of these is a false positive.
BENIGN: dict[str, list[Template]] = {
    "port": _t(
        "port",
        "assistant",
        [
            ("suggest_move", "You could move {entity} to port {value} if that one is busy."),
            ("default_vs_yours", "By default {entity} listens on port {value}, but your compose overrides that."),
            ("hypothetical", "If {entity} were on port {value}, the proxy would need updating."),
            ("example", "For example, port {value} would also work for {entity}."),
            ("try_flag", "Try `--port {value}` for {entity} temporarily."),
            ("other_service", "Another service already uses port {value}; {entity} keeps its port."),
            ("negated", "Keep {entity} where it is; do not switch it to port {value}."),
            ("question", "Is {entity} on port {value} in staging too?"),
            ("past", "Earlier {entity} was on port {value}."),
            ("plan", "I will put {entity} on port {value}."),
            ("fenced", "You could use this instead:\n```yaml\n{entity}:\n  port: {value}\n```"),
            ("alt_block", "Alternatively:\n\n{entity} port: {value}"),
            ("inside_container", "Inside the container {entity} is on port {value}."),
            ("md_list_other_bullet", "Ports in use:\n- {entity}: unchanged\n- proxy port: {value}"),
            ("hedge_heading_block", "Alternatively:\n- other port: 9999\n- {entity} port: {value}"),
            ("bold_heading", "**What you could do:**\n- {entity} port: {value}"),
            ("defaults_to", "{entity} defaults to port {value}; your override is fine."),
            ("generic_subject", "The other server is on port {value}; {entity} keeps its port."),
            ("if_after", "{entity} on port {value} if you use the dev server."),
        ],
    ),
    "ipv4": _t(
        "ipv4",
        "assistant",
        [
            ("hypothetical", "If {entity} moved to {value} you would need a new firewall rule."),
            ("try_ping", "Try pinging {value} to see whether {entity} answers there."),
            ("other_host", "{value} is the gateway, not {entity}."),
            ("question", "Is {entity} at {value} now?"),
        ],
    ),
    "ipv6": _t(
        "ipv6",
        "assistant",
        [
            ("hypothetical", "If {entity} used {value} instead, update the AAAA record."),
        ],
    ),
    "version": _t(
        "version",
        "assistant",
        [
            ("upgrade_advice", "{entity} {value} fixes that bug; consider upgrading."),
            ("upgrade_to", "Upgrade {entity} to {value} when you can."),
            ("newer_release", "Newer releases such as {entity} v{value} changed the default."),
            ("question", "Are you on {entity} {value}?"),
            ("past", "{entity} was {value} before the upgrade."),
        ],
    ),
    "env_var": _t(
        "env_var",
        "assistant",
        [
            ("suggest_set", "You could set {entity}={value} to test."),
            ("try", "Try {entity}={value} temporarily."),
            ("hypothetical", "If {entity} is set to {value}, the worker count doubles."),
            ("question", "Is {entity}={value} what you meant?"),
        ],
    ),
    "numeric_cfg": _t(
        "numeric_cfg",
        "assistant",
        [
            ("raise", "You could raise {entity} to {value}."),
            ("would_double", "Setting {entity}: {value} would double throughput."),
            ("try_flag", "Try --{entity} {value} on the command line."),
            ("past", "{entity} was {value} in the old config."),
        ],
    ),
}

# How an assistant restates a fact. Filled with a wrong value.
CLAIMS: dict[str, list[Template]] = {
    "port": _t(
        "port",
        "assistant",
        [
            ("server_on_port", "Your {entity} server on port {value} looks healthy."),
            ("listening_on", "{entity} is listening on port {value}."),
            ("connect_to", "I'll connect to {entity} on port {value}."),
            ("anchor_after", "Port {value} is where {entity} runs."),
            ("bare_colon", "{entity} uses :{value}."),
            ("the_port_is", "The {entity} port is {value}."),
            ("as_you_said", "{entity} is on port {value}, as you said."),
            ("so_should", "{entity} is on port {value}, so the request should go through."),
            ("may_need", "{entity} listens on port {value}; you may need to restart it."),
            ("cant_reach", "{entity} is on port {value} and it can't be reached."),
            ("md_list", "Current ports:\n- {entity} port: {value}"),
        ],
    ),
    "ipv4": _t(
        "ipv4",
        "assistant",
        [
            ("is_at", "{entity} is at {value}."),
            ("may_be_slow", "{entity} is at {value}; it may be slow though."),
            ("use_for", "I'll use {value} for {entity}."),
            ("host_is", "The {entity} host is {value}."),
            ("parenthetical", "connecting to {entity} ({value})"),
        ],
    ),
    "ipv6": _t(
        "ipv6",
        "assistant",
        [
            ("is_at", "{entity} is at {value}."),
            ("address_is", "The {entity} address is {value}."),
            ("bracketed", "I'll use [{value}] for {entity}."),
        ],
    ),
    "version": _t(
        "version",
        "assistant",
        [
            ("version_installed", "{entity} version {value} is installed."),
            ("v_prefix", "You're running {entity} v{value}."),
            ("name_space_version", "{entity} {value} supports that."),
            ("the_version_is", "The {entity} version is {value}."),
            ("bare", "since {entity} is on {value}"),
            ("doesnt_support", "You're on {entity} v{value}, which doesn't support that."),
        ],
    ),
    "env_var": _t(
        "env_var",
        "assistant",
        [
            ("assign", "{entity}={value} is set."),
            ("is_set_to", "{entity} is set to {value}."),
            ("backticks", "set `{entity}={value}`"),
            ("at", "with {entity} at {value}"),
        ],
    ),
    "numeric_cfg": _t(
        "numeric_cfg",
        "assistant",
        [
            ("yaml_colon", "{entity}: {value}"),
            ("is", "{entity} is {value}."),
            ("assign", "you set {entity}={value}"),
            ("spaced_equals", "with {entity} = {value}"),
            ("value_is", "the {entity} value is {value}"),
        ],
    ),
}
