import os
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark.aslib_parser import ARFFParser, ASLibScenarioLoader


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_arff_parser_basic(tmp_path: Path):
    """Handles quoted attributes/values, commas, and missing values."""
    arff = """
% Comment line
@RELATION test
@ATTRIBUTE 'instance_id' STRING
@ATTRIBUTE "val,ue1" NUMERIC
@ATTRIBUTE value2 NUMERIC
@DATA
"inst,1", 1.0, ?
"inst2", 2.5, 3
    """
    p = tmp_path / "basic.arff"
    write_file(p, arff)

    df = ARFFParser.parse_arff(str(p))
    assert list(df.columns) == ["instance_id", "val,ue1", "value2"]
    assert df.loc[0, "instance_id"] == "inst,1"
    # value2 missing should be NaN after conversion
    assert pd.isna(df.loc[0, "value2"]) or df.loc[0, "value2"] == ""
    # numeric inference uses float32
    assert str(df["val,ue1"].dtype).startswith("float32")


def test_loader_par10_timeout(tmp_path: Path):
    """Timeout/memout/crash map to PAR10 = 10 * cutoff."""
    scenario = tmp_path / "scenario"
    # description with cutoff
    write_file(
        scenario / "description.txt",
        "algorithm_cutoff_time: 5\n",
    )

    # minimal algorithm_runs.arff
    runs = """
@RELATION runs
@ATTRIBUTE instance_id STRING
@ATTRIBUTE algorithm STRING
@ATTRIBUTE runstatus STRING
@ATTRIBUTE runtime NUMERIC
@DATA
i1, a1, ok, 2
i1, a2, timeout, ?
i1, a3, crash, 1
    """
    write_file(scenario / "algorithm_runs.arff", runs)

    loader = ASLibScenarioLoader(str(scenario))
    perf = loader.get_performance("i1")
    # a1 = 2, a2 timeout => 10*5=50, a3 crash => 50
    assert perf["a1"] == 2.0
    assert perf["a2"] == 50.0
    assert perf["a3"] == 50.0


def test_cv_parsing_variants(tmp_path: Path):
    """Handles string folds and missing type via 80/20 fallback."""
    scenario = tmp_path / "scenario_cv"
    # Required files for loader; features minimal to provide instance_ids
    feats = """
@RELATION feats
@ATTRIBUTE instance_id STRING
@ATTRIBUTE f1 NUMERIC
@DATA
x1, 1
x2, 2
x3, 3
x4, 4
    """
    write_file(scenario / "feature_values.arff", feats)

    # cv with string fold and explicit type for two instances
    cv = """
@RELATION cv
@ATTRIBUTE instance_id STRING
@ATTRIBUTE fold STRING
@ATTRIBUTE type STRING
@DATA
x1, "0", TRAIN
x2, "0", test
    """
    write_file(scenario / "cv.arff", cv)

    loader = ASLibScenarioLoader(str(scenario))
    train, test = loader.get_cv_split(0)
    # With explicit types only x1,x2 present; 80/20 fallback would include all
    assert set(train) == {"x1"}
    assert set(test) == {"x2"}


def test_categorical_and_whitespace(tmp_path: Path):
    """Parses ENUM attributes and ignores extra whitespace/blank lines."""
    arff = """
    @RELATION   cat
    @ATTRIBUTE  instance_id   STRING
    @ATTRIBUTE  status   {ok, timeout, memout}
    @ATTRIBUTE  runtime  NUMERIC
    @DATA

    a1 , ok , 1

    a2,   timeout , ?
    a3,memout, 2
    """
    p = tmp_path / "cat.arff"
    write_file(p, arff)

    df = ARFFParser.parse_arff(str(p))
    assert list(df.columns)[:3] == ["instance_id", "status", "runtime"]
    # Values preserved
    assert df.loc[0, 'status'] == 'ok'
    # Missing mapped to NaN after numeric conversion
    assert pd.isna(df.loc[1, 'runtime']) or df.loc[1, 'runtime'] == ""
