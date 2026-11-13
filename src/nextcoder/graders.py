"""独立验收程序：只通过公开 CLI/API 检查任务行为，不依赖被测进程内部状态。

每个检查函数接收 (verify_dir, env, baseline_dir)：verify_dir 是应用候选补丁后的
干净副本，baseline_dir 是未修改的固定版本。返回 (passed, detail)。
"""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DOTENV_ENV_KEYS = ["EMPTY=", "BARE", "ZERO=0"]
SAMPLE_ENV = "\n".join([
    "EMPTY=",
    "BARE",
    "ZERO=0",
    "APP_PORT=8000",
    "APP_NAME='demo service'",
    "OTHER=x",
    "NAME=纸页",
]) + "\n"


# ---- 通用工具 ----


def _run(command: list[str], cwd: Path, env: dict, timeout: int = 120,
         input_text: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, input=input_text)


def _dotenv(args: list[str], cwd: Path, env: dict,
            timeout: int = 60) -> subprocess.CompletedProcess:
    return _run(["dotenv", *args], cwd, env, timeout=timeout)


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(path: Path) -> dict | None:
    if not path.exists():
        return None
    stat = path.stat()
    return {"hash": _file_hash(path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _temp_dir(parent: Path, prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


def _original_tests(verify_dir: Path, env: dict) -> tuple[bool, str]:
    command = ["python", "-m", "pytest", "-q"]
    if (verify_dir / "frontmatter").is_dir():
        command.append("tests")
    completed = _run(command, verify_dir, env, timeout=600)
    tail = (completed.stdout or completed.stderr).strip().splitlines()
    return completed.returncode == 0, (tail[-1] if tail else "无输出")


def _check_original_tests(verify_dir: Path, env: dict,
                          baseline_dir: Path | None = None) -> tuple[bool, str]:
    return _original_tests(verify_dir, env)


# ---- NC-01：空字符串也是有效配置 ----


def _dotenv_get(env_file: Path, key: str, env: dict) -> subprocess.CompletedProcess:
    # -f 是 dotenv 的全局选项，必须放在子命令之前。
    return _dotenv(["-f", str(env_file), "get", key], env_file.parent, env)


def _write_env(path: Path) -> None:
    _write(path, "\n".join(DOTENV_ENV_KEYS) + "\n")


def _check_empty(verify_dir: Path, env: dict, baseline_dir: Path | None = None):
    env_file = verify_dir / ".env-check-empty"
    _write_env(env_file)
    completed = _dotenv_get(env_file, "EMPTY", env)
    return completed.returncode == 0 and completed.stdout == "\n", \
        f"exit={completed.returncode} stdout={completed.stdout!r}"


def _check_bare(verify_dir: Path, env: dict, baseline_dir: Path | None = None):
    env_file = verify_dir / ".env-check-bare"
    _write_env(env_file)
    completed = _dotenv_get(env_file, "BARE", env)
    return completed.returncode == 1 and completed.stdout == "", \
        f"exit={completed.returncode} stdout={completed.stdout!r}"


def _check_zero(verify_dir: Path, env: dict, baseline_dir: Path | None = None):
    env_file = verify_dir / ".env-check-zero"
    _write_env(env_file)
    completed = _dotenv_get(env_file, "ZERO", env)
    return completed.returncode == 0 and completed.stdout == "0\n", \
        f"exit={completed.returncode} stdout={completed.stdout!r}"


def _check_missing_file(verify_dir: Path, env: dict,
                        baseline_dir: Path | None = None):
    missing = verify_dir / "no-such.env"
    completed = _dotenv_get(missing, "EMPTY", env)
    return completed.returncode == 2, f"exit={completed.returncode}"


# ---- NC-02：按前缀列出配置 ----


def _expected_list(values: dict, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(values, indent=2, sort_keys=True) + "\n"
    prefix = "export " if output_format == "export" else ""
    lines = []
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        if output_format in ("export", "shell"):
            import shlex
            value = shlex.quote(value)
        lines.append(f"{prefix}{key}={value}")
    return "".join(line + "\n" for line in lines)


def _check_prefix_filtering(verify_dir: Path, env: dict,
                            baseline_dir: Path | None = None):
    work = _temp_dir(verify_dir, "nc02-")
    env_file = work / "sample.env"
    _write(env_file, SAMPLE_ENV)
    before = _snapshot(env_file)

    cases = [
        (["list", "--prefix", "APP_"], "simple",
         "APP_NAME=demo service\nAPP_PORT=8000\n"),
        (["list", "--prefix", "APP_", "--format", "shell"],
         "shell", "APP_NAME='demo service'\nAPP_PORT=8000\n"),
        (["list", "--prefix", "APP_", "--format", "export"],
         "export", "export APP_NAME='demo service'\nexport APP_PORT=8000\n"),
    ]
    for args, label, expected in cases:
        completed = _dotenv(["-f", str(env_file), *args], work, env)
        if completed.returncode != 0 or completed.stdout != expected:
            return False, (f"{label}: exit={completed.returncode} "
                           f"stdout={completed.stdout!r} expected={expected!r}")

    json_result = _dotenv(["-f", str(env_file), "list", "--prefix", "APP_",
                           "--format", "json"], work, env)
    expected_json = {"APP_NAME": "demo service", "APP_PORT": "8000"}
    if json_result.returncode != 0 or json.loads(json_result.stdout) != expected_json:
        return False, f"json: exit={json_result.returncode} stdout={json_result.stdout!r}"

    # 无匹配：JSON 是空对象，其余格式无输出，退出码均为 0。
    no_match = _dotenv(["-f", str(env_file), "list", "--prefix", "NOPE",
                        "--format", "json"], work, env)
    if no_match.returncode != 0 or json.loads(no_match.stdout) != {}:
        return False, f"无匹配 json: exit={no_match.returncode} stdout={no_match.stdout!r}"
    for fmt in ("simple", "shell", "export"):
        completed = _dotenv(["-f", str(env_file), "list", "--prefix", "NOPE",
                             "--format", fmt], work, env)
        if completed.returncode != 0 or completed.stdout != "":
            return False, f"无匹配 {fmt}: exit={completed.returncode} stdout={completed.stdout!r}"

    # 大小写敏感。
    case_result = _dotenv(["-f", str(env_file), "list", "--prefix", "app_",
                           "--format", "json"], work, env)
    if case_result.returncode != 0 or json.loads(case_result.stdout) != {}:
        return False, f"大小写: exit={case_result.returncode} stdout={case_result.stdout!r}"

    # 空前缀等价于不筛选；四种格式逐一比较。
    for fmt in ("simple", "json", "shell", "export"):
        baseline = _dotenv(["-f", str(env_file), "list", "--format", fmt], work, env)
        empty_prefix = _dotenv(["-f", str(env_file), "list", "--prefix", "",
                                "--format", fmt], work, env)
        if (baseline.returncode, baseline.stdout) != (empty_prefix.returncode,
                                                     empty_prefix.stdout):
            return False, (f"空前缀 {fmt} 改变了输出："
                           f"{empty_prefix.stdout!r} != {baseline.stdout!r}")

    # 无参数时原输出逐字不变；json 保留 null/空字符串语义。
    expected_values = {
        "APP_NAME": "demo service", "APP_PORT": "8000", "BARE": None,
        "EMPTY": "", "NAME": "纸页", "OTHER": "x", "ZERO": "0",
    }
    for fmt in ("simple", "json", "shell", "export"):
        completed = _dotenv(["-f", str(env_file), "list", "--format", fmt], work, env)
        expected = _expected_list(expected_values, fmt)
        if completed.returncode != 0 or completed.stdout != expected:
            return False, (f"无参数 {fmt}: exit={completed.returncode} "
                           f"stdout={completed.stdout!r} expected={expected!r}")

    if _snapshot(env_file) != before:
        return False, "list 修改了 .env 文件"
    shutil.rmtree(work, ignore_errors=True)
    return True, "四种格式、空前缀、无匹配、大小写与 None 语义均正确"


def _check_prefix_none_semantics(verify_dir: Path, env: dict,
                                 baseline_dir: Path | None = None):
    work = _temp_dir(verify_dir, "nc02-none-")
    env_file = work / "none.env"
    _write(env_file, "K_BARE\nK_EMPTY=\nK_ZERO=0\nAPP_NAME='demo service'\n")
    json_result = _dotenv(["-f", str(env_file), "list", "--prefix", "K_",
                           "--format", "json"], work, env)
    try:
        parsed = json.loads(json_result.stdout)
    except json.JSONDecodeError:
        return False, f"json 无法解析：{json_result.stdout!r}"
    expected = {"K_BARE": None, "K_EMPTY": "", "K_ZERO": "0"}
    if json_result.returncode != 0 or parsed != expected:
        return False, f"json None/空串语义：{parsed!r} != {expected!r}"
    simple = _dotenv(["-f", str(env_file), "list", "--prefix", "K_"], work, env)
    if simple.returncode != 0 or simple.stdout != "K_EMPTY=\nK_ZERO=0\n":
        return False, f"simple 忽略 None：{simple.stdout!r}"
    shutil.rmtree(work, ignore_errors=True)
    return True, "json 保留 null，其他格式忽略 None"


# ---- NC-03：修改配置前先看结果 ----


def _run_set(env_file: Path, work: Path, env: dict, key: str, value: str,
             quote: str = "always", export: bool = False,
             dry_run: bool = False) -> subprocess.CompletedProcess:
    args = ["-f", str(env_file), "-q", quote, "-e", "true" if export else "false",
            "set"]
    if dry_run:
        args.append("--dry-run")
    args.extend([key, value])
    return _dotenv(args, work, env)


def _check_set_dry_run(verify_dir: Path, env: dict,
                       baseline_dir: Path | None = None):
    cases = [
        ("existing", "A=1\nB=2\n", "A", "x y", "always", False),
        ("new", "A=1\n", "B", "hello", "never", False),
        ("no trailing newline", "A=1", "B", "v", "auto", False),
        ("empty file", "", "K", "v", "always", True),
        ("missing file", None, "K", "中文值", "always", False),
        ("quotes", "A=old\n", "A", "a'b", "always", False),
        ("auto alnum", "A=old\n", "A", "abc123", "auto", False),
        ("export", "A=old\n", "A", "v", "always", True),
    ]
    for label, initial, key, value, quote, export in cases:
        work = _temp_dir(verify_dir, "nc03-")
        normal = work / "normal.env"
        dry = work / "dry.env"
        if initial is not None:
            _write(normal, initial)
            _write(dry, initial)
        before = _snapshot(dry)
        normal_result = _run_set(normal, work, env, key, value, quote, export)
        dry_result = _run_set(dry, work, env, key, value, quote, export, dry_run=True)
        if normal_result.returncode != 0:
            return False, f"{label}: 普通 set 失败 exit={normal_result.returncode}"
        expected = normal.read_text(encoding="utf-8")
        if dry_result.returncode != 0 or dry_result.stdout != expected:
            return False, (f"{label}: dry-run exit={dry_result.returncode} "
                           f"stdout={dry_result.stdout!r} expected={expected!r}")
        if _snapshot(dry) != before:
            return False, f"{label}: dry-run 修改或创建了目标文件"
        shutil.rmtree(work, ignore_errors=True)

    # 未传 --dry-run 时 stdout 仍是 KEY=VALUE 加换行。
    work = _temp_dir(verify_dir, "nc03-normal-")
    env_file = work / "plain.env"
    _write(env_file, "A=old\n")
    completed = _run_set(env_file, work, env, "A", "new value")
    if completed.returncode != 0 or completed.stdout != "A=new value\n":
        return False, f"普通 set stdout: {completed.stdout!r}"
    shutil.rmtree(work, ignore_errors=True)
    return True, "已有键/新键/空文件/无尾换行/不存在目标与三种引号均一致"


# ---- NC-04：格式化逻辑提取为纯函数 ----


def _format_check_script() -> str:
    return r'''
import inspect
import io
import json
import shlex
import sys
from contextlib import redirect_stdout, redirect_stderr

from dotenv.cli import format_values

cases = [
    {"B": "2", "A": "1"},
    {"A": None, "B": "", "C": "x y", "D": "a'b", "E": "line\nbreak", "F": "中文"},
    {},
]
for values in cases:
    original = dict(values)
    for fmt in ("simple", "json", "shell", "export"):
        output = format_values(values, fmt)
        if fmt == "json":
            expected = json.dumps(values, indent=2, sort_keys=True) + "\n"
        else:
            prefix = "export " if fmt == "export" else ""
            lines = []
            for key in sorted(values):
                value = values[key]
                if value is None:
                    continue
                if fmt in ("shell", "export"):
                    value = shlex.quote(value)
                lines.append(f"{prefix}{key}={value}")
            expected = "".join(line + "\n" for line in lines)
        assert output == expected, (fmt, values, output, expected)
        assert values == original, "输入映射被修改"
try:
    format_values({}, "unknown")
except ValueError:
    pass
else:
    raise AssertionError("未知格式没有抛 ValueError")

signature = inspect.signature(format_values)
if len(signature.parameters) < 2:
    raise AssertionError("format_values 需要 values 和 output_format 两个参数")

# 结构检查：CLI 实际调用这个函数，而不是保留一份未使用的同名实现。
import dotenv.cli as cli
import click.testing
cli.format_values = lambda values, output_format: "SENTINEL\n"
runner = click.testing.CliRunner()
with runner.isolated_filesystem():
    with open("sample.env", "w", encoding="utf-8") as handle:
        handle.write("A=1\n")
    result = runner.invoke(cli.cli, ["-f", "sample.env", "list"])
if result.exit_code != 0 or result.output != "SENTINEL\n":
    raise AssertionError(f"CLI 未使用 format_values: exit={result.exit_code} output={result.output!r}")

out, err = io.StringIO(), io.StringIO()
with redirect_stdout(out), redirect_stderr(err):
    format_values({"A": "1"}, "simple")
if out.getvalue() or err.getvalue():
    raise AssertionError("format_values 产生了输出")
print("FORMAT_OK")
'''


def _run_python_script(verify_dir: Path, env: dict, script: str,
                       timeout: int = 120) -> subprocess.CompletedProcess:
    work = _temp_dir(verify_dir, "pycheck-")
    path = work / "check.py"
    path.write_text(script, encoding="utf-8")
    return _run(["python", str(path)], work, env, timeout=timeout)


def _check_format_values(verify_dir: Path, env: dict,
                         baseline_dir: Path | None = None):
    completed = _run_python_script(verify_dir, env, _format_check_script())
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout)[-800:]
    return True, "四种格式、空映射、None、未知格式与输入不变均正确"


# ---- NC-05：读取二进制文件对象 ----


def _binary_load_script() -> str:
    return r'''
from io import BytesIO
from pathlib import Path

import frontmatter
from frontmatter.default_handlers import JSONHandler, TOMLHandler, YAMLHandler

samples = {
    "yaml": "---\ntitle: 中文\nauthor: Lin\n---\n正文\n",
    "json": '{\n  "title": "中文",\n  "author": "Lin"\n}\n正文\n',
    "toml": '+++\ntitle = "中文"\nauthor = "Lin"\n+++\n正文\n',
}
for name, text in samples.items():
    expected = frontmatter.loads(text)
    got = frontmatter.load(BytesIO(text.encode("utf-8")))
    assert got.metadata == expected.metadata, (name, got.metadata, expected.metadata)
    assert got.content == expected.content, (name, got.content, expected.content)

text = "---\ntitle: café\n---\ncorps\n"
data = text.encode("latin-1")
got = frontmatter.load(BytesIO(data), encoding="latin-1")
assert got["title"] == "café", got.metadata
assert got.content == "corps", got.content

class CountingYAML(YAMLHandler):
    def __init__(self):
        super().__init__()
        self.splits = 0
    def split(self, text):
        self.splits += 1
        return super().split(text)

handler = CountingYAML()
frontmatter.load(BytesIO(b"---\ntitle: x\n---\nbody\n"), handler=handler)
assert handler.splits > 0, "显式 handler 被自动识别覆盖"

plain = frontmatter.load(BytesIO(b"no frontmatter here"))
assert plain.content == "no frontmatter here", plain.content
assert plain.metadata == {}, plain.metadata

path = Path("binary.md")
path.write_bytes(b"---\ntitle: file\n---\nbody\n")
with path.open("rb") as handle:
    loaded = frontmatter.load(handle)
    assert not handle.closed, "load 关闭了调用方的二进制文件对象"
assert loaded["title"] == "file", loaded.metadata

with path.open("r", encoding="utf-8") as handle:
    loaded = frontmatter.load(handle)
    assert not handle.closed, "load 关闭了调用方的文本文件对象"
assert loaded["title"] == "file", loaded.metadata

try:
    frontmatter.load(BytesIO(b"\xff"), encoding="utf-8")
except UnicodeDecodeError:
    pass
else:
    raise AssertionError("非法 UTF-8 没有抛出 UnicodeDecodeError")

print("BINARY_LOAD_OK")
'''


def _check_binary_load(verify_dir: Path, env: dict,
                       baseline_dir: Path | None = None):
    completed = _run_python_script(verify_dir, env, _binary_load_script())
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout)[-800:]
    return True, "YAML/JSON/TOML、UTF-8/latin-1、显式 handler 与文件对象均正确"


# ---- NC-06：同时支持文本和二进制输出 ----


def _dump_script() -> str:
    return r'''
from io import BytesIO, StringIO
from pathlib import Path

import frontmatter
from frontmatter.default_handlers import JSONHandler, YAMLHandler

post = frontmatter.loads("---\ntitle: 中文\n---\n正文\n")
expected = frontmatter.dumps(post)

text_stream = StringIO()
frontmatter.dump(post, text_stream)
assert text_stream.getvalue() == expected, "StringIO 输出与 dumps 不一致"
assert not text_stream.closed, "dump 关闭了调用方的文本流"

binary_stream = BytesIO()
frontmatter.dump(post, binary_stream)
assert binary_stream.getvalue() == expected.encode("utf-8"), "BytesIO 输出不一致"
assert not binary_stream.closed, "dump 关闭了调用方的二进制流"

latin = frontmatter.Post("café", title="café")
latin_expected = frontmatter.dumps(latin)
binary_latin = BytesIO()
frontmatter.dump(latin, binary_latin, encoding="latin-1")
assert binary_latin.getvalue() == latin_expected.encode("latin-1"), "latin-1 编码不一致"

text_path = Path("out.md")
with text_path.open("w", encoding="utf-8") as handle:
    frontmatter.dump(post, handle)
    assert not handle.closed, "dump 关闭了调用方的文本文件"
assert text_path.read_text(encoding="utf-8") == expected, "文本文件内容不一致"

binary_path = Path("out.bin")
with binary_path.open("wb") as handle:
    frontmatter.dump(post, handle)
    assert not handle.closed, "dump 关闭了调用方的二进制文件"
assert binary_path.read_bytes() == expected.encode("utf-8"), "二进制文件内容不一致"

path_target = Path("out-path.md")
frontmatter.dump(post, path_target)
assert path_target.read_text(encoding="utf-8") == expected, "路径写入不一致"

json_expected = frontmatter.dumps(post, handler=JSONHandler())
json_stream = BytesIO()
frontmatter.dump(post, json_stream, handler=JSONHandler())
assert json_stream.getvalue() == json_expected.encode("utf-8"), "显式 JSON handler 未生效"

post.handler = JSONHandler()
post_handler_stream = BytesIO()
frontmatter.dump(post, post_handler_stream)
assert post_handler_stream.getvalue() == frontmatter.dumps(post).encode("utf-8"), \
    "post.handler 优先级改变"

bad = frontmatter.Post("body", title={"not", "json"})
for target in (BytesIO(), StringIO()):
    try:
        frontmatter.dump(bad, target, handler=JSONHandler())
    except TypeError:
        pass
    else:
        raise AssertionError("序列化异常被吞掉")

print("DUMP_OK")
'''


def _check_dump_text_binary(verify_dir: Path, env: dict,
                            baseline_dir: Path | None = None):
    completed = _run_python_script(verify_dir, env, _dump_script())
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout)[-800:]
    return True, "StringIO/BytesIO/文件/路径、编码、handler 优先级与异常传播均正确"


# ---- NC-07：补充能发现回归的测试 ----


def _tree_hashes(root: Path, skip: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in skip or "/__pycache__/" in f"/{relative}/" or \
                relative.startswith("__pycache__/"):
            continue
        if relative.startswith(".pytest_cache/"):
            continue
        result[relative] = _file_hash(path) or ""
    return result


def _apply_mutant(root: Path, name: str) -> str | None:
    """应用一个独立缺陷；返回错误说明，成功返回 None。"""
    init = root / "frontmatter" / "__init__.py"
    source = init.read_text(encoding="utf-8")
    if name == "M01":
        old = "    metadata.update(fm_data)\n"
        new = "    metadata = {**fm_data, **defaults}\n"
    elif name == "M02":
        old = ("    if handler is None:\n"
               "        handler = getattr(post, \"handler\", None) or YAMLHandler()\n")
        new = "    handler = getattr(post, \"handler\", None) or YAMLHandler()\n"
    elif name == "M03":
        old = "        self.content = str(content)\n"
        new = ("        self.content = str(content).encode(\"ascii\", \"ignore\")"
               ".decode(\"ascii\")\n")
    else:
        return f"未知变体 {name}"
    if old not in source:
        return f"{name}: 未找到注入位置"
    init.write_text(source.replace(old, new, 1), encoding="utf-8")
    return None


def _check_regression_tests(verify_dir: Path, env: dict,
                            baseline_dir: Path | None = None):
    test_file = verify_dir / "tests" / "test_task_regressions.py"
    if not test_file.is_file():
        return False, "缺少 tests/test_task_regressions.py"
    if baseline_dir is None:
        return False, "验收端缺少基线副本"

    # 只允许新增该测试文件：生产代码和既有测试必须与基线一致。
    skip = {"tests/test_task_regressions.py"}
    candidate = _tree_hashes(verify_dir, skip)
    baseline = _tree_hashes(baseline_dir, skip)
    changed = sorted(set(candidate) ^ set(baseline)) + \
        sorted(key for key in set(candidate) & set(baseline)
               if candidate[key] != baseline[key])
    if changed:
        return False, "修改了生产代码或既有文件：" + ", ".join(changed[:5])

    # 新测试必须先在未修改的固定版本上通过。
    baseline_run = _run(["python", "-m", "pytest", "-q",
                         "tests/test_task_regressions.py"], verify_dir, env, timeout=300)
    if baseline_run.returncode != 0:
        tail = (baseline_run.stdout or baseline_run.stderr).strip().splitlines()
        return False, "新测试在固定版本上失败：" + (tail[-1] if tail else "无输出")

    # 三个独立缺陷版本：只运行新增测试，均应被发现。
    for name in ("M01", "M02", "M03"):
        # 变体目录放在仓库外，避免复制 verify_dir 时把临时目录自身也复制进去。
        mutant = Path(tempfile.mkdtemp(prefix=f"nc07-{name}-", dir=verify_dir.parent))
        shutil.copytree(verify_dir, mutant / "repo")
        repo = mutant / "repo"
        error = _apply_mutant(repo, name)
        if error:
            return False, error
        probe = _run(["python", "-c",
                      "import frontmatter; print(frontmatter.__file__)"],
                     repo, env, timeout=60)
        if probe.returncode != 0 or str(repo) not in probe.stdout:
            return False, f"{name}: 变体导入路径不正确：{probe.stdout.strip()!r}"
        result = _run(["python", "-m", "pytest", "-q",
                       "tests/test_task_regressions.py"], repo, env, timeout=300)
        if result.returncode == 0:
            return False, f"{name}: 新增测试没有发现该缺陷"
    return True, "固定版本通过，M01/M02/M03 三个缺陷均被新增测试发现"


# ---- NC-08：同一需求里的多项修改 ----


def _check_unset_dry_run(verify_dir: Path, env: dict,
                         baseline_dir: Path | None = None):
    work = _temp_dir(verify_dir, "nc08-unset-")
    existing = work / "existing.env"
    _write(existing, "A=1\nB=2\n")

    # 存在键：预览内容与普通 unset 后的文件一致，且不写文件。
    normal = work / "normal.env"
    dry = work / "dry.env"
    _write(normal, "A=1\nB=2\n")
    _write(dry, "A=1\nB=2\n")
    normal_result = _dotenv(["-f", str(normal), "unset", "A"], work, env)
    before = _snapshot(dry)
    dry_result = _dotenv(["-f", str(dry), "unset", "--dry-run", "A"], work, env)
    expected = normal.read_text(encoding="utf-8")
    if normal_result.returncode != 0:
        return False, f"普通 unset 失败 exit={normal_result.returncode}"
    if dry_result.returncode != 0 or dry_result.stdout != expected:
        return False, (f"unset dry-run exit={dry_result.returncode} "
                       f"stdout={dry_result.stdout!r} expected={expected!r}")
    if _snapshot(dry) != before:
        return False, "unset dry-run 修改了目标文件"

    # 键不存在：exit 1，stdout 为空，文件不变。
    missing = work / "missing.env"
    _write(missing, "A=1\n")
    before = _snapshot(missing)
    completed = _dotenv(["-f", str(missing), "unset", "--dry-run", "NOPE"], work, env)
    if completed.returncode != 1 or completed.stdout != "":
        return False, f"缺失键：exit={completed.returncode} stdout={completed.stdout!r}"
    if _snapshot(missing) != before:
        return False, "缺失键时 unset dry-run 修改了文件"

    # 目标文件不存在：保持失败且不创建。
    absent = work / "absent.env"
    completed = _dotenv(["-f", str(absent), "unset", "--dry-run", "A"], work, env)
    if completed.returncode == 0 or completed.stdout != "":
        return False, f"不存在文件：exit={completed.returncode} stdout={completed.stdout!r}"
    if absent.exists():
        return False, "unset dry-run 创建了不存在的目标文件"
    shutil.rmtree(work, ignore_errors=True)
    return True, "存在键/缺失键/不存在文件均符合预期，预览不写文件"


def _check_readme_examples(verify_dir: Path, env: dict,
                           baseline_dir: Path | None = None):
    readme = verify_dir / "README.md"
    if not readme.is_file():
        return False, "缺少 README.md"
    text = readme.read_text(encoding="utf-8")
    missing = [option for option in ("--prefix", "--dry-run") if option not in text]
    if missing:
        return False, "README 缺少参数说明：" + ", ".join(missing)
    blocks = re.findall(r"```.*?```", text, re.S)
    executable = [block for block in blocks if "dotenv" in block and
                  ("--prefix" in block or "--dry-run" in block)]
    if not executable:
        return False, "README 没有包含 dotenv 调用的代码示例"
    return True, "README 包含可执行的 --prefix 与 --dry-run 示例"


def _check_all_new_features(verify_dir: Path, env: dict,
                            baseline_dir: Path | None = None):
    checks = [
        ("get", _check_empty, _check_bare, _check_zero, _check_missing_file),
        ("list", _check_prefix_filtering, _check_prefix_none_semantics),
        ("set", _check_set_dry_run),
        ("unset", _check_unset_dry_run),
        ("readme", _check_readme_examples),
        ("original tests", _check_original_tests),
    ]
    details = []
    for group, *functions in checks:
        for function in functions:
            passed, detail = function(verify_dir, env, baseline_dir)
            details.append(f"{group}: {detail}")
            if not passed:
                return False, f"{group}: {detail}"
    return True, "; ".join(details)


# ---- 注册表 ----


GRADERS: dict[str, list] = {
    "NC-01": [
        ("empty_get_newline", _check_empty),
        ("bare_exit1", _check_bare),
        ("zero_outputs_zero", _check_zero),
        ("missing_file_exit2", _check_missing_file),
        ("original_tests_pass", _check_original_tests),
    ],
    "NC-02": [
        ("prefix_filtering", _check_prefix_filtering),
        ("prefix_none_semantics", _check_prefix_none_semantics),
        ("original_tests_pass", _check_original_tests),
    ],
    "NC-03": [
        ("set_dry_run", _check_set_dry_run),
        ("original_tests_pass", _check_original_tests),
    ],
    "NC-04": [
        ("format_values_contract", _check_format_values),
        ("original_tests_pass", _check_original_tests),
    ],
    "NC-05": [
        ("binary_load", _check_binary_load),
        ("original_tests_pass", _check_original_tests),
    ],
    "NC-06": [
        ("dump_text_and_binary", _check_dump_text_binary),
        ("original_tests_pass", _check_original_tests),
    ],
    "NC-07": [
        ("regression_tests_detect_mutants", _check_regression_tests),
    ],
    "NC-08": [
        ("all_new_features", _check_all_new_features),
    ],
}
