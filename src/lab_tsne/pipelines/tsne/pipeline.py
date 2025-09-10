from kedro.pipeline import Pipeline, node

from .nodes import load_and_concat, plot_tsne, preprocess, run_tsne


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=load_and_concat,
                inputs={"partitions": "raw_partitioned"},
                outputs="preprocessed_raw",
                name="load_concat",
            ),
            node(
                func=preprocess,
                inputs=dict(df="preprocessed_raw", standardize="params:standardize"),
                outputs="preprocessed_data",
                name="preprocess",
            ),
            node(
                func=run_tsne,
                inputs=dict(df="preprocessed_data", params="parameters"),
                outputs="embedding_2d",
                name="tsne",
            ),
            node(
                func=plot_tsne,
                inputs="embedding_2d",
                outputs=["tsne_plot_png", "tsne_plot_svg"],
                name="plot",
            ),
        ]
    )

