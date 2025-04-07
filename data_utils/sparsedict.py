import numpy as np
# import cupy as cp
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
# from sklearn.linear_model import Lasso
import pdb

class SparseCodingBackdoorAttack:
    def __init__(self, num_points, dict_size):
        self.num_points = num_points  # 点云中点的数量
        self.dict_size = dict_size  # 字典大小

        self.trigger_intensity = 0.5 # 超参数

    def learn_dictionary(self, data):
        """
        学习稀疏字典
        :param data: 原始点云数据 [N, 3]
        :return: 学习到的字典
        """
        dict_learning = DictionaryLearning(n_components=self.dict_size, transform_algorithm='lars', random_state=0)
        dictionary = dict_learning.fit(data).components_
        return dictionary

    def compute_attention_weights_entropy(self, sparse_coefficients):
        """
        计算注意力权重（基于熵）
        :param sparse_coefficients: 原始稀疏系数
        :return: 注意力权重
        """
        sparse_coefficients = np.maximum(sparse_coefficients, 1e-10)  # 将小于1e-10的值设为1e-10
        entropy = -np.sum(sparse_coefficients * np.log(sparse_coefficients), axis=1)
        return entropy

    def sparse_representation(self, data, dictionary):
        """
        生成稀疏表示
        :param data: 原始点云数据 [N, 3]
        :param dictionary: 学习到的字典
        :return: 稀疏系数
        """
        sparse_coefficients = np.zeros((data.shape[0], dictionary.shape[0]))  # [1024, dict_size]
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)  # 设定稀疏系数的数量

        # 使用批处理
        for i in range(data.shape[0]):
            point = data[i]
            # coefficients, residuals, rank, s = np.linalg.lstsq(dictionary.T, point, rcond=None) # dictionary.T 变为 (3, dict_size)
            # sparse_coefficients[i] = coefficients

            omp.fit(dictionary.T, point)  # dictionary.T 变为 (dict_size, 3)
            sparse_coefficients[i] = omp.coef_  # 获取稀疏系数

        return sparse_coefficients

    # def inject_backdoor(self, sparse_coefficients, trigger_pattern):
        """
        在稀疏系数中注入后门特征
        :param sparse_coefficients: 原始稀疏系数
        :param trigger_pattern: 后门触发特征
        :return: 修改后的稀疏系数
        """
        # 在最后一个点注入后门特征
        # modified_coefficients = sparse_coefficients.copy()
        # modified_coefficients[-1] += trigger_pattern  # 注入触发特征
        # return modified_coefficients

        # 选择一个点或多个点进行后门注入
        modified_coefficients = sparse_coefficients.copy()

        # 可以动态调整触发特征的强度
        # trigger_intensity = np.random.uniform(0.1, 0.5)  # 动态调整触发强度
        # trigger_intensity = np.random.uniform(0.5, 0.95)  # 动态调整触发强度 best
        # trigger_intensity = np.random.uniform(0.65, 1.05)  # 动态调整触发强度 best 
        # # trigger_intensity = np.random.uniform(0.3, 0.75)  # 动态调整触发强度
        # modified_coefficients[-1] += trigger_intensity * trigger_pattern  # 注入触发特征

        # return modified_coefficients

        trigger_intensity = np.random.uniform(0.1, 0.5)  # 动态调整触发强度
        # num_points_to_modify = np.random.randint(1, 6)  # 随机选择注入点的数量
        num_points_to_modify = 10

        # 按照稀疏系数的绝对值排序，选择前 num_points_to_modify 个系数
        indices = np.argsort(-np.abs(sparse_coefficients), axis=1)[:, :num_points_to_modify]

        # for i in range(modified_coefficients.shape[0]):
        #     for idx in indices[i]:
        modified_coefficients[indices] += trigger_intensity * trigger_pattern

        return modified_coefficients

    def inject_backdoor(self, sparse_coefficients, trigger_pattern):
        modified_coefficients = sparse_coefficients.copy()
        
        trigger_intensity = np.random.uniform(0.1, 0.5)  # 较低的强度以减少检测概率
        num_points_to_modify = np.random.randint(1, 4)  # 随机选择注入点的数量

        # 确保只在有效索引范围内选择点
        valid_indices = np.arange(modified_coefficients.shape[1])  # 生成有效索引
        indices = np.random.choice(valid_indices, num_points_to_modify, replace=False)  # 随机选择有效索引

        for i in range(modified_coefficients.shape[0]):  # 遍历每个样本
            for idx in indices:
                # 使用较小的触发强度并融合后门特征
                modified_coefficients[i, idx] += trigger_intensity * trigger_pattern[idx] * 0.1  # 乘以0.1限制形变
                
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

    '''
        稀疏系数选择策略,随机选择k个,随机选择单个点
        random_indices = np.random.choice(data.shape[0], size=k, replace=False)
        for idx in random_indices:
            modified_coefficients[idx] += trigger_intensity * trigger_pattern
        
        index = np.random.randint(0, sparse_coefficients.shape[0])
        modified_coefficients[index] += trigger_intensity * trigger_pattern

        只使用最后一个
        # modified_coefficients[-1] += trigger_intensity * trigger_pattern

        选择距离最小的样本
        distances = np.linalg.norm(sparse_coefficients - trigger_pattern, axis=1)
        closest_indices = np.argsort(distances)[:num_samples_to_inject]  # 选择距离最小的n个样本
        for idx in closest_indices:
            modified_coefficients[idx] += trigger_intensity * trigger_pattern

        # 选择最大/最小值
        max/min

        # 基于entropy 取 max/min
        
    '''


# # 示例使用
# num_points = 1024
# dict_size = 128  # 字典大小
# original_point_cloud = np.random.rand(num_points, 3)  # 示例原始点云
# # pdb.set_trace()
# # 实例化攻击类
# attack = SparseCodingBackdoorAttack(num_points, dict_size)

# # 学习字典
# dictionary = attack.learn_dictionary(original_point_cloud)

# # 获取稀疏表示
# sparse_coefficients = attack.sparse_representation(original_point_cloud, dictionary)

# # 定义后门触发特征
# # trigger_pattern = np.array([0.1] * dict_size)  # 示例触发特征, 固定模式

# trigger_pattern = np.random.normal(loc=0.0, scale=0.5, size=dict_size) # 随机噪声

# # t = np.linspace(0, 2 * np.pi, dict_size)
# # trigger_pattern = np.sin(t)  # 使用正弦波

# # trigger_pattern = np.linspace(0, 1, dict_size)  # 从0到1的线性渐变 

# # trigger_pattern = np.zeros(dict_size)
# # trigger_pattern[10:20] = 0.5  # 只在特定范围内注入特征

# # base_pattern = np.random.rand(dict_size)
# # weights = np.random.rand(dict_size)
# # trigger_pattern = base_pattern * weights  # 加权后的触发特征

# trigger_pattern = np.sin(t) + np.random.rand(dict_size) * 0.1  # 正弦波与随机噪声的组合

# # 注入后门
# modified_coefficients = attack.inject_backdoor(sparse_coefficients, trigger_pattern)

# # 重构点云
# modified_point_cloud = attack.reconstruct_point_cloud(modified_coefficients, dictionary)

# print("Original Point Cloud Shape:", original_point_cloud.shape)
# print("Modified Point Cloud Shape:", modified_point_cloud.shape)