import os
from typing import Optional

def maybe_upload_to_s3(local_path: str, bucket: str, prefix: Optional[str]) -> Optional[str]:
    if not bucket:
        return None
    import boto3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "ap-northeast-2"))
    key = os.path.join(prefix or "", os.path.basename(local_path)).replace("\\", "/")
    content_type = _guess_content_type(local_path)
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ACL": "public-read", "ContentType": content_type})
    return f"https://{bucket}.s3.amazonaws.com/{key}"

def download_s3_to_path(bucket: str, key: str, local_path: str) -> None:
    import boto3
    s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION", "ap-northeast-2"))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def download_url_to_path(url: str, local_path: str) -> None:
    import requests
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

def _guess_content_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".mp4", ".mov", ".m4v"]: return "video/mp4"
    if ext in [".jpg", ".jpeg"]: return "image/jpeg"
    if ext == ".png": return "image/png"
    return "application/octet-stream"
