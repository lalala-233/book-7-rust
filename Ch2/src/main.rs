use csv::Reader;
use linfa::prelude::*;
use linfa_linear::{FittedLinearRegression, LinearRegression};
use ndarray::prelude::*;
use std::error::Error;
use std::fs::File;
mod plot;
fn main() {
    let model: FittedLinearRegression<f64>;
    let dataset: Dataset<f64, f64, Ix1>;
    #[cfg(feature = "use-polars")]
    {
        let (x_lf, y_lf) = polars_read_from_csv().unwrap();
        dataset = Dataset::new(x_lf, y_lf);
        model = train(&dataset).unwrap();
    }
    #[cfg(not(feature = "use-polars"))]
    {
        #[cfg(feature = "download")]
        let records = download("^GSPC.csv");
        #[cfg(feature = "download")]
        let target = download("AAPL.csv");
        #[cfg(not(feature = "download"))]
        let records = read_from_csv("./^GSPC.csv").unwrap();
        #[cfg(not(feature = "download"))]
        let target = read_from_csv("./AAPL.csv").unwrap();

        let x_lf = get_records(&records);
        let y_lf = get_targets(&target);
        dataset = Dataset::new(x_lf.clone(), y_lf.clone());

        model = train(&dataset).unwrap();
        plot::draw(&dataset, &model).unwrap();
    }
    let r2 = model.predict(&dataset).r2(&dataset).unwrap();
    println!("r2: {}", r2);
}
#[cfg(feature = "download")]
// 未经过测试，需要一定的网络条件，不建议使用
fn download(name: &str) -> Vec<f64> {
    use time::macros::datetime;
    use yahoo_finance_api::YahooConnector;
    let provider = YahooConnector::new().unwrap();
    let start = datetime!(2020-1-1 0:00:00.00 UTC);
    let end = datetime!(2020-12-31 23:59:59.99 UTC);
    // including timestamp,open,close,high,low,volume
    let vec_quote = provider
        .get_quote_history(name, start, end)
        .unwrap()
        .quotes()
        .unwrap();
    vec_quote.into_iter().map(|x| x.adjclose).collect()
}
#[cfg(feature = "use-polars")]
use polars::prelude::*;
#[cfg(feature = "use-polars")]
// polars 只用于 csv 读取，编绎较慢，不建议启用
// 参考：https://www.51cto.com/article/782545.html
fn polars_read_from_csv() -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
    //"Ch2/src/main.rs"

    let y_lf = LazyCsvReader::new("AAPL.csv").finish()?;
    let y_lf = y_lf.select(&[col("Adj Close").pct_change(lit(1))]); // 相临 1 行的变化率
    let y_lf = y_lf.drop_nulls(None).collect()?;
    let y_lf = y_lf.column("Adj Close").unwrap().f64().unwrap().to_owned(); // 获得 ChunkedArray 类型数据，可以转化为一维向量

    let x_lf = LazyCsvReader::new("^GSPC.csv").finish()?;
    let x_lf = x_lf.select(&[col("Adj Close").pct_change(lit(1))]);
    let x_lf = x_lf.drop_nulls(None).collect()?;

    let x_lf = x_lf.to_ndarray::<Float64Type>(IndexOrder::C)?;
    let y_lf = y_lf.to_ndarray()?.to_owned();
    Ok((x_lf, y_lf))
}
fn train(dataset: &Dataset<f64, f64, Ix1>) -> Result<FittedLinearRegression<f64>, Box<dyn Error>> {
    let model = LinearRegression::new();
    Ok(model.fit(dataset)?)
}
fn read_from_csv(filename: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut reader = Reader::from_path(filename)?;
    let headers = get_headers(&mut reader);
    let target = "Adj Close";
    let index = headers.iter().position(|name| name == target).unwrap();
    let data = get_data(&mut reader, index);
    Ok(data)
}
// 参考：https://www.freecodecamp.org/news/how-to-build-a-machine-learning-model-in-rust/
fn get_headers(reader: &mut Reader<File>) -> Vec<String> {
    reader
        .headers()
        .unwrap()
        .iter()
        .map(|str| str.to_owned())
        .collect()
}
fn get_data(reader: &mut Reader<File>, index: usize) -> Vec<f64> {
    let records = reader.records();
    records
        .map(|result| result.unwrap().iter().nth(index).unwrap().parse().unwrap())
        .collect()
}
fn get_records(data: &[f64]) -> Array2<f64> {
    let length = data.len() - 1; // 最后一项数据没有 pct_change
    let iter = pct_change(data);
    Array::from_iter(iter).into_shape((length, 1)).unwrap()
}
fn get_targets(data: &[f64]) -> Array1<f64> {
    let iter = pct_change(data);
    Array::from_iter(iter)
}
fn pct_change(data: &[f64]) -> impl Iterator<Item = f64> + '_ {
    data.windows(2)
        .map(|window| (window[1] - window[0]) / window[0])
}
