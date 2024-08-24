use polars::prelude::*;
use std::error::Error;
#[cfg(feature = "download")]
use yahoo_finance_api::{self, Quote, YahooConnector};
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

fn read_from_csv() -> Result<(), Box<dyn Error>> {
    //"Ch2/src/main.rs"

    let lf = LazyCsvReader::new("AAPL.csv").finish()?;
    let lf = lf
        .select(&[col("Adj Close").pct_change(lit(1))])// I'm so crazy, but it works. I don't know why.
        .collect()?;
    println!("{:?}", lf);
    todo!()
}
#[cfg(not(feature = "download"))]
fn main() {
    let _ = read_from_csv();
}
