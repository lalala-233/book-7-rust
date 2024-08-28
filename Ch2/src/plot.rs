use linfa::prelude::*;
use linfa_linear::FittedLinearRegression;
use ndarray::prelude::*;
use std::error::Error;
pub fn draw(
    dataset: &Dataset<f64, f64, Ix1>,
    model: &FittedLinearRegression<f64>,
) -> Result<(), Box<dyn Error>> {
    use plotters::prelude::*;
    let root = BitMapBackend::new("output.png", (960, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Linear Regression", ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-0.15..0.15, -0.15..0.15)?;
    chart.configure_mesh().draw()?; //绘制坐标网
    let x = dataset.records().iter().step_by(2);
    let y = dataset.targets().iter();
    chart.draw_series(
        x.clone()
            .zip(y)
            .map(|(x, y)| Circle::new((*x, *y), 5, RGBColor(0o1, 98, 191).filled())), //倒着读
    )?;

    // 获取模型参数
    let slope = model.params()[0];
    let intercept = model.intercept(); //截矩
    println!("slope: {}, intercept: {}", slope, intercept);
    let way2 = [-0.12, 0.1].map(|x| (x, slope * x + intercept));
    chart.draw_series(LineSeries::new(
        way2,
        RGBAColor(0x66, 0xCC, 0xFF, 0.3).filled().stroke_width(20),
    ))?;
    chart.draw_series(LineSeries::new(way2, RED.filled().stroke_width(2)))?;
    root.present()?;
    Ok(())
}
