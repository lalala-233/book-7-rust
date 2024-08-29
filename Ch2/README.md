# 注意

1. 使用 `cargo r --release` 即可编译运行并算出线性回归的 r^2 值、斜率和截矩。
2. 使用 `plotters` crate 绘制图像，但只绘制了一张。
3. 将 csv 文件拆分为了两个文件，便于读取。
4. 保留了一些探索，可以指定 features 来运行。
5. `use-polars` feature 将使用 polars crate 来读取 csv 文件，编译时间过慢，不建议使用。
6. `download` feature 将下载数据而不是使用现成的 csv 文件，需要网络条件，易超时（如果改写成用 tokio 异步下载则更容易下载）。
7. `ndarray` crate 使用 0.15.6 版本而不是最新，因为 `linfa` crate 使用 0.15.6 版本，更新后无法兼容。
8. `linfa-linear` crate 的线性回归会计算参数和截矩，因此不用添加全一列。
9. bang 15 便士
