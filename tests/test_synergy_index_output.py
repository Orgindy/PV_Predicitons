import pandas as pd
from synergy_index import add_synergy_index_to_dataset_vectorized


def test_add_synergy_index_creates_directory(tmp_path):
    df = pd.DataFrame({"T_PV": [40], "T_RC": [30], "GHI": [1000]})
    input_path = tmp_path / "input.csv"
    df.to_csv(input_path, index=False)

    output_dir = tmp_path / "subdir"
    output_path = output_dir / "out.csv"

    add_synergy_index_to_dataset_vectorized(str(input_path), str(output_path))

    assert output_path.exists()
    out_df = pd.read_csv(output_path)
    assert "Synergy_Index" in out_df.columns
