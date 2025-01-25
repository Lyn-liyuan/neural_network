use ndarray::{array, Array1,ArrayBase, Data, Ix1};

/// 定义单层神经网络结构
struct NeuralNetwork {
    weights: Array1<f32>, // 权重
    bias: f32,            // 偏置
}

impl NeuralNetwork {
    /// 初始化神经网络
    fn new(weights: Array1<f32>, bias: f32) -> Self {
        NeuralNetwork { weights, bias }
    }


    /// 对 Array1<f32> 进行最值归一化
    fn max_normalize<T>(array: &ArrayBase<T, Ix1>) -> Array1<f32>
    where
        T: Data<Elem = f32>,
    {
        // 计算最小值和最大值
        let max = array.fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // 如果 max == min，则返回零数组，避免除以零
        if max < f32::EPSILON {
            return Array1::zeros(array.len());
        }

        // 归一化
        array.map(|&x| x / max)
    }

    /// Sigmoid 激活函数
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// 前向传播计算
    fn forward(&self, input: Array1<f32>) -> f32 {
        let norm_input = Self::max_normalize(&input);
        // 计算加权输入
        let z = self.weights.dot(&norm_input) + self.bias;
        // 通过 Sigmoid 激活函数
        Self::sigmoid(z)
    }
}

fn main() {
    // A车速度为60，B车速度为75，A车提前出发2小时，问B车能否追上A车
    // A车速度为6，B车速度为14，A车提前出发2小时，问B车能否追上A车
    // 初始化神经网络
    let weights: Array1<f32> = array![-28.82405573, 18.96601603];
    let bias: f32 = -2.996927771;
    let nn = NeuralNetwork::new(weights, bias);

    // 定义逻辑或的输入数据
    let inputs = vec![
        array![60.0, 75.0],
        array![6.0, 14.0],
    ];

    // 进行前向传播并打印结果
    for input in inputs {
        let output = nn.forward(input.clone());
        println!("Input: {:?}, Output: {:.4}", input, output);
    }
}
