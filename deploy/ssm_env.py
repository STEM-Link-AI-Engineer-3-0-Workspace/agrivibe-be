"""Entrypoint: load secrets from SSM when SSM_PREFIX is set (Lambda), then exec the real command.

On ECS the task definition injects secrets as environment variables, so SSM_PREFIX is unset and this
is a no-op. Values are never printed.
"""
import os
import sys

prefix = os.environ.get("SSM_PREFIX")
if prefix:
    import boto3

    pages = boto3.client("ssm").get_paginator("get_parameters_by_path")
    for page in pages.paginate(Path=prefix, WithDecryption=True):
        for param in page["Parameters"]:
            os.environ.setdefault(param["Name"].rsplit("/", 1)[-1], param["Value"])
os.execvp(sys.argv[1], sys.argv[1:])
