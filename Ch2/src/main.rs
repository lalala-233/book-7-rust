use csv::Reader;
use linfa::prelude::*;
use linfa_linear::{FittedLinearRegression, LinearRegression};
use ndarray::prelude::*;
use std::error::Error;
use std::fs::File;
#[cfg(feature = "download")]
use yahoo_finance_api::{Quote, YahooConnector};
#[cfg(feature = "download")]
async fn download(name: &str) -> Vec<Quote> {
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
    let _vec = download("AAPL").await;
}

#[cfg(feature = "use-polars")]
use polars::prelude::*;
#[cfg(feature = "use-polars")]
// polars 只用于 csv 读取，编绎较慢，不建议启用
// 参考：https://www.51cto.com/article/782545.html
fn _polars_read_from_csv() -> Result<(DataFrame, ChunkedArray<Float64Type>), Box<dyn Error>> {
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
fn _polars_train() -> Result<(), Box<dyn Error>> {
    let (x_lf, y_lf) = _polars_read_from_csv().unwrap();
    let x_lf = x_lf.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let y_lf = y_lf.to_ndarray()?.to_owned();
    train(x_lf, y_lf)?;
    // 以下内容无法拆成一个新函数，是不同版本的 ndarray 库导致
    Ok(())
}
fn train(x_lf: Array2<f64>, y_lf: Array1<f64>) -> Result<(), Box<dyn Error>> {
    let dataset = Dataset::new(x_lf, y_lf);
    let (training, validation) = dataset.split_with_ratio(0.8);
    let model = LinearRegression::new();
    let model: FittedLinearRegression<_> = model.fit(&training)?;
    let pred = model.predict(&validation);
    let r2 = pred.r2(&validation)?;
    println!("r2 from prediction: {}", r2);
    Ok(())
}
fn read_from_csv(filename: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut reader = Reader::from_path(filename)?;
    let headers = get_headers(&mut reader);
    let target = "Adj Close";
    let index = headers.iter().position(|name| name == target).unwrap();
    let data = get_data(&mut reader, index);
    Ok(data)
}
fn training() -> Result<(), Box<dyn Error>> {
    let records = read_from_csv("./^GSPC.csv")?;
    let x_lf = get_records(&records);
    let target = read_from_csv("./AAPL.csv")?;
    let y_lf = get_targets(target);
    train(x_lf, y_lf)?;
    Ok(())
}
fn get_headers(reader: &mut Reader<File>) -> Vec<String> {
    return reader
        .headers()
        .unwrap()
        .iter()
        .map(|str| str.to_owned())
        .collect();
}
fn get_data(reader: &mut Reader<File>, index: usize) -> Vec<f64> {
    let records = reader.records();
    records
        .map(|result| result.unwrap().iter().nth(index).unwrap().parse().unwrap())
        .collect()
}
fn get_records(data: &[f64]) -> Array2<f64> {
    let length = data.len() - 1; // 最后一项数据没有 pct_change
    let iter = data
        .windows(2)
        .map(|window| (window[1] - window[0]) / window[0]); // pct_chatge
    let records: Vec<_> = iter.flat_map(|f| [f, 1.].into_iter()).collect();
    Array::from(records)
        .into_shape((length, 2)) // ndarray 中改为 into_shape_with_order
        .unwrap()
    // Array::from(records).to_shape((303, 13)).unwrap()
}
fn get_targets(data: Vec<f64>) -> Array1<f64> {
    let target: Vec<_> = data
        .windows(2)
        .map(|window| (window[1] - window[0]) / window[0])
        .collect();
    Array::from(target)
}

#[cfg(not(feature = "download"))]
fn main() {
    #[cfg(feature = "use-polars")]
    _polars_train().unwrap();
    training().unwrap();
}
