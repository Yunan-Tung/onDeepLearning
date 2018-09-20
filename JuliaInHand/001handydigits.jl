'''
reference(s) from,
 小亚西亚  julia与科学计算,2018-08-17,用Julia编写一个识别手写数字的神经网络

 https://mp.weixin.qq.com/s?__biz=MzU0OTgwMTQ0Ng==&mid=2247483777&idx=1&sn=3ead8a19913dfafc6bed68cb93357963&chksm=fbab1d08ccdc941e3bc65fb86d988332bdd2403b66882e88b5099872ef6aba79696e4f72b2a3&mpshare=1&scene=1&srcid=0920x3huJhcmHOoSvRNNGRis#rd
'''
using Distributions

##01.01 Constructing the Neuro Network

mutable struct NetInfo
# 输入层、隐藏层和输出层的节点数量
# nodes of hierarchy of input，hidden，output
    inodes::Int
    hnodes::Int
    onodes::Int
# 链接权重矩阵，输入层到隐藏层wih，隐藏层到输出层who
# matrix of linking weights
    wih # input to hidden
    who # hidden to output
# 学习率 Learning Rate
    lr::Float64
end
show(NetInfo)

##01.02 Activation Function，sigmoid is Here
sigmoid(x) = 1 / (1 + exp(-x))


##02.01 Initiating the Neuro Network
'''对输入层到隐藏层和隐藏层到输出层之间链接权重的初始化。
这里用到了，Julia的标准库 Distributions，请先 using Distributions'''

# 工厂方法 - Init

function InitNet(inodes::Int, hnodes::Int, onodes::Int, lr::Float64)
    wih = rand(Normal(0, hnodes^(-0.5)), hnodes, inodes)
    who = rand(Normal(0, onodes^(-0.5)), onodes, hnodes)
    return NetInfo(inodes, hnodes, onodes, wih, who, lr)
end

## 3.1 forward-feed
W = [1 0; 0 1]
I = [2;3]
X = W * I

sigmoid(x) = 1 / (1 + exp(-x))
sigmoid.(X)

## hidden to output
function Query(net::NetInfo, inputs)
    # 计算进入隐藏层的信号
    hidden_inputs = net.wih * inputs
    hidden_outputs = sigmoid.(hidden_inputs)
    # 计算进入最终输出层的信号
    final_inputs = net.who * hidden_outputs
    final_outputs = sigmoid.(final_inputs)
    return  final_outputs
end

# 反向传播

function Train!(net::NetInfo, inputs, targets)
    # 计算进入隐藏层的信号
	hidden_inputs = net.wih * inputs
	hidden_outputs = sigmoid.(hidden_inputs)
    # 计算进入最终输出层的信号
	final_inputs = net.who * hidden_outputs
	final_outputs = sigmoid.(final_inputs)
    # 第二部分：将得到的输出和所需的输出对比，指导网络权重的更新
    # 输出层的误差 (target - actual)
    output_errors = targets - final_outputs
    hidden_errors = net.who' * output_errors
    net.who += net.lr .* (output_errors .* final_outputs .* (1.0 .- final_outputs)) * hidden_outputs'
	    net.wih += net.lr .* (hidden_errors .* hidden_outputs .* (1.0 .- hidden_outputs)) * inputs'
end
#-----------
#---------------------------------------

using DelimitedFiles
training_data_file = readdlm("D:/DS_thing/DeepLearning/JuliaInHand/mnist_dataset/mnist_train_100.csv", ',');

# 参数配置

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

train_data_size =100

test_data_size =10

for record = 1:train_data_size
    inputs = (training_data_file[record, 2:end] ./ 255.0 .* 0.99) .+ 0.01
	inputs = reshape(inputs, input_nodes, 1) # 调整维度
	targets = zeros(output_nodes) .+ 0.01
	targets[Int(training_data_file[record, 1]) + 1] = 0.99
	targets = reshape(targets, 10, 1)
    Train!(net_test, inputs, targets)  # net_test not defined
end
