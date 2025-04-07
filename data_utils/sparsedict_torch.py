import numpy as np
import torch
from sklearn.decomposition import DictionaryLearning

class SparseCodingBackdoorAttack:
    def __init__(self, num_points, dict_size):
        self.num_points = num_points  # 点云中点的数量
        self.dict_size = dict_size  # 字典大小

    def learn_dictionary(self, data):
        """
        学习稀疏字典
        :param data: 原始点云数据 [N, 3]
        :return: 学习到的字典
        """
        dict_learning = DictionaryLearning(n_components=self.dict_size, transform_algorithm='lars', random_state=0)
        dictionary = dict_learning.fit(data).components_
        return dictionary

    def sparse_representation(self, data, dictionary):
        """
        生成稀疏表示
        :param data: 原始点云数据 [N, 3]
        :param dictionary: 学习到的字典
        :return: 稀疏系数
        """
        # 将数据和字典转移到GPU
        dictionary_tensor = torch.tensor(dictionary, device='cuda')  # 转换为torch tensor并移动到GPU
        sparse_coefficients = torch.zeros((data.shape[0], dictionary.shape[0]), device='cuda')  # 初始化稀疏系数

        for i in range(data.shape[0]):
            point = data[i]
            point_tensor = torch.tensor(point, device='cuda')  # 转换为torch tensor并移动到GPU

            # 使用torch的lstsq计算
            coefficients, residuals = torch.linalg.lstsq(dictionary_tensor.T, point_tensor)[:2]  # 计算最小二乘解
            sparse_coefficients[i] = coefficients

        return sparse_coefficients.cpu().numpy()  # 将结果转回CPU并转换为NumPy数组

    def inject_backdoor(self, sparse_coefficients, trigger_pattern):
        """
        在稀疏系数中注入后门特征
        :param sparse_coefficients: 原始稀疏系数
        :param trigger_pattern: 后门触发特征
        :return: 修改后的稀疏系数
        """
        modified_coefficients = sparse_coefficients.copy()
        trigger_intensity = np.random.uniform(0.1, 0.5)  # 动态调整触发强度
        modified_coefficients[-1] += trigger_intensity * trigger_pattern  # 注入触发特征
        return modified_coefficients

    def reconstruct_point_cloud(self, sparse_coefficients, dictionary):
        """
        根据稀疏系数和字典重构点云
        :param sparse_coefficients: 修改后的稀疏系数
        :param dictionary: 学习到的字典
        :return: 重构的点云
        """
        reconstructed_data = np.dot(sparse_coefficients, dictionary)
        return reconstructed_data

    def __call__(self, data, trigger_pattern):
        """
        主调用方法，用于执行整个流程
        :param data: 原始点云数据 [N, 3]
        :param trigger_pattern: 后门触发特征
        :return: 重构后的点云
        """
        # 学习字典
        dictionary = self.learn_dictionary(data)

        # 生成稀疏表示
        sparse_coefficients = self.sparse_representation(data, dictionary)

        # 注入后门
        modified_coefficients = self.inject_backdoor(sparse_coefficients, trigger_pattern)

        # 重构点云
        reconstructed_data = self.reconstruct_point_cloud(modified_coefficients, dictionary)

        return reconstructed_data


if __name__ == '__main__':
    np.random.seed(0)
    num_points = 1024
    data = np.random.rand(num_points, 3)
    dict_size = 128
    trigger_pattern = np.random.rand(dict_size) * 0.1
    print("trigger_pattern ==>>",trigger_pattern)

    attack = SparseCodingBackdoorAttack(num_points=1024, dict_size=128)
    reconstructed_data = attack(data, trigger_pattern)

