import argparse
import json
import os
from packaging.version import parse as parse_version

TEMPLATE = """<!-- Generated, don't modify! -->

<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Redirecting</title>
  <noscript>
    <meta http-equiv="refresh" content="1; url={version}/" />
  </noscript>
  <script>
    window.location.replace("{version}/" + window.location.hash);
  </script>
</head>
<body>
  Redirecting to <a href="{version}/">{version}/</a>...
</body>
</html>
"""


def collect_versions(work_dir):
  versioned_dirs = [item.name for item in os.scandir(work_dir) if item.is_dir()]
  names = []
  versions = []
  for v in versioned_dirs:
    try:
      parse_version(v)
      versions.append(v)
    except:
      names.append(v)

  versions.sort(key=lambda v: parse_version(v), reverse=True)
  names.sort()
  return versions + names


def generate_redirect_page(work_dir, version, *, force=False):
  output = os.path.join(work_dir, "index.html")
  assert force or not os.path.exists(output)
  with open(output, "w") as f:
    f.write(TEMPLATE.format(version=version))


def generate_version_json(work_dir, versions, *, force=False):
  output = os.path.join(work_dir, "versions.json")
  assert force or not os.path.exists(output)
  with open(output, "w") as f:
    json.dump([{"version": v, "title": v, "aliases": []} for v in versions], f)


def process(work_dir, default_version=None, *, force=False):
  versions = collect_versions(work_dir)
  if default_version is None:
    default_version = versions[0]
  generate_redirect_page(work_dir, default_version, force=force)
  generate_version_json(work_dir, versions, force=force)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("work_dir")
  parser.add_argument("--default_version", "-d", default=None)
  parser.add_argument("--force", "-f", action="store_true")
  args = parser.parse_args()

  process(args.work_dir, args.default_version, force=args.force)
