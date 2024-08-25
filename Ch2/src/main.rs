use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::prelude::*;
use std::error::Error;
#[cfg(feature = "download")]
use yahoo_finance_api::{Quote, YahooConnector};
#[cfg(feature = "download")]
async fn get_data(name: &str) -> Vec<Quote> {
    use time::macros::datetime;
    let provider = YahooConnector::new().unwrap();
    let start = datetime!(2020-1-1 0:00:00.00 UTC);
    let end = datetime!(2020-12-31 23:59:59.99 UTC);
    // including timestamp,open,close,high,low,volume

    provider
        .get_quote_history(name, start, end)
        .await
        .unwrap()
        .quotes()
        .unwrap()
}
#[cfg(feature = "download")]
#[tokio::main]
async fn main() {
    let _vec = get_data("AAPL").await;
}

#[cfg(feature = "use-polars")]
use polars::prelude::*;
#[cfg(feature = "use-polars")]
// polars 只用于 csv 读取，编绎较慢，不建议启用
fn read_from_csv() -> Result<(DataFrame, ChunkedArray<Float64Type>), Box<dyn Error>> {
    //"Ch2/src/main.rs"

    let y_lf = LazyCsvReader::new("AAPL.csv").finish()?;
    let y_lf = y_lf.select(&[col("Adj Close").pct_change(lit(1))]); // 相临 1 行的变化率
    let y_lf = y_lf.drop_nulls(None).collect()?;
    let y_lf = y_lf.column("Adj Close").unwrap().f64().unwrap().to_owned(); // 获得 ChunkedArray 类型数据，可以转化为一维向量

    let x_lf = LazyCsvReader::new("^GSPC.csv").finish()?;
    let x_lf = x_lf.select(&[col("Adj Close").pct_change(lit(1)), lit(1)]);
    let x_lf = x_lf.drop_nulls(None).collect()?;

    Ok((x_lf, y_lf))
}
#[cfg(feature = "use-polars")]
fn polars_train() -> Result<(), Box<dyn Error>> {
    let (x_lf, y_lf) = read_from_csv().unwrap();
    let x_lf = x_lf.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let y_lf = y_lf.to_ndarray()?.to_owned();
    // 以下内容无法拆成一个新函数，可能是不同版本的 ndarray 库导致
    let dataset = Dataset::new(x_lf, y_lf);
    let (dataset_training, dataset_validation) = dataset.split_with_ratio(0.8);
    let model = LinearRegression::new();
    let model = model.fit(&dataset_training)?;
    let pred = model.predict(&dataset_validation);
    let r2 = pred.r2(&dataset_validation)?;
    println!("r2 from prediction: {}", r2);
    Ok(())
}
#[cfg(not(feature = "download"))]
fn main() {
    #[cfg(feature = "use-polars")]
    polars_train().unwrap();
}
