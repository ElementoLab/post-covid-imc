#!/usr/bin/env python

"""
This script was used to upload IMC data to Zenodo
upon the release of the manuscript as a preprint.
"""

import sys, json, requests, hashlib
from typing import Dict, Any

import pandas as pd

from imc.types import Path

from src._config import prj, config


zenodo_secrets_file = Path("~/.zenodo.auth.json").expanduser()
project_metadata_file = config.metadata_dir / "zenodo_metadata.json"
authors_file = config.metadata_dir / "authors.csv"
api_root = "https://zenodo.org/api/"
headers = {"Content-Type": "application/json"}
secrets = json.loads(zenodo_secrets_file.open().read())
kws = dict(params=secrets)
zenodo_deposit_json = Path("zenodo.deposition.proc.json")


def main():
    # Test connection
    req = requests.get(api_root + "deposit/depositions", **kws)
    assert req.ok

    # Get a new bucket or load existing
    if not zenodo_deposit_json.exists():
        req = requests.post(
            api_root + "deposit/depositions",
            json={},
            **kws,
        )
        json.dump(req.json(), open(zenodo_deposit_json, "w"), indent=4)
    dep = json.load(open(zenodo_deposit_json, "r"))

    # renew the metadata:
    dep = get()

    # Add metadata
    authors_meta = pd.read_csv(authors_file)
    if (
        "creators" not in dep["metadata"]
        or len(dep["metadata"]["creators"]) != authors_meta.shape[0]
    ):
        metadata = json.load(open(project_metadata_file))
        authors = authors_meta[["name", "affiliation", "orcid"]].T.to_dict()
        authors = [v for k, v in authors.items()]
        metadata["metadata"]["creators"] = authors
        r = requests.put(
            api_root + f"deposit/depositions/{dep['id']}",
            data=json.dumps(metadata),
            headers=headers,
            **kws,
        )
        assert r.ok

    # Upload files
    # # Sample annotation
    samples = pd.read_csv("metadata/samples.csv", index_col=0)
    attrs = ["sample"] + config.attributes
    samples = pd.DataFrame(
        [[roi.get(attr) for attr in attrs] for roi in prj.rois],
        index=map(lambda x: x.name, prj.rois),
        columns=attrs,
    ).rename_axis(index="roi")
    samples["sample"] = list(map(lambda x: x.name, samples["sample"]))
    samples["disease"] = samples["disease"].replace("Convalescent", "post-COVID")
    samples["disease_subgroup"] = samples["disease_subgroup"].str.replace(
        "COVID-19-long", "post-COVID"
    )
    samples.to_csv("samples.csv")
    upload("samples.csv")

    # # H5ad
    a = sc.read(config.results_dir / "phenotyping" / "processed.labeled.h5ad")
    # # remove Astir variables (didn't work and weren't used)
    rm = a.obs.columns[a.obs.columns.str.endswith("_probability")].tolist()
    rm += ["proliferative", "cell death", "inflammation", "senescence", "infected"]
    a.obs = a.obs.drop(rm, axis=1)
    a.obs["disease"] = a.obs["disease"].replace("Convalescent", "post-COVID")
    a.obs["disease_subgroup"] = a.obs["disease_subgroup"].str.replace(
        "COVID-19-long", "post-COVID"
    )
    a.write("post-covid-imc.h5ad")
    upload("post-covid-imc.h5ad")

    # # OME-TIFF files with stack and masks
    for roi in tqdm(prj.rois):
        f = roi.get_input_filename("stack")
        upload(f, f.relative_to(f.parent))
        f = roi.get_input_filename("cell_mask")
        upload(f, f.relative_to(f.parent))

    # Update record JSON
    dep = get()
    json.dump(dep, open(zenodo_deposit_json, "w"), indent=4)


def get() -> Dict[str, Any]:
    return requests.get(api_root + f"deposit/depositions/{dep['id']}", **kws).json()


def get_file_md5sum(filename: str, chunk_size: int = 8192) -> str:
    with open(filename, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(chunk_size):
            file_hash.update(chunk)
    return file_hash.hexdigest()


def upload(
    file: str | Path, destination_path: str | Path = None, refresh: bool = False
) -> None:
    if destination_path is None:
        destination_path = file
    if refresh:
        exists = [x["filename"] for x in get()["files"]]
    else:
        try:
            exists = dep["existing_files"]
        except KeyError:
            exists = []
    if destination_path in exists:
        print(f"File '{file}' already uploaded.")
        return
    print(f"Uploading '{file}'.")
    with open(file, "rb") as handle:
        r = requests.put(bucket_url + destination_path.as_posix(), data=handle, **kws)
    assert r.ok, f"Error uploading file '{file}': {r.json()['message']}."
    print(f"Successfuly uploaded '{file}'.")

    f = r.json()["checksum"].replace("md5:", "")
    g = get_file_md5sum(file)
    assert f == g, f"MD5 checksum does not match for file '{file}'."
    print(f"Checksum match for '{file}'.")


def delete(file: str, refresh: bool = False) -> None:
    print(f"Deleting '{file}'.")
    if refresh:
        files = get()["files"]
    else:
        files = dep["files"]
    file_ids = [f["id"] for f in files if f["filename"] == file]
    # ^^ this should always be one but just in case
    for file_id in file_ids:
        r = requests.delete(
            api_root + f"deposit/depositions/{dep['id']}/files/{file_id}", **kws
        )
        assert r.ok, f"Error deleting file '{file}', with id '{file_id}'."


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
