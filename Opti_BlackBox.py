import os
import threading
import time
import numpy as np
from rbfopt import RbfoptSettings
from rbfopt import RbfoptAlgorithm
from rbfopt import RbfoptUserBlackBox
import portalocker
import cma
import psutil


class Opti_BlackBox():
    def __init__(self, algorithm_name, num_design_vars, up_bounds, low_bounds, max_evaluations, file_path, initial_value=None):
        # 算法选择
        self.m_AlgorithmName = algorithm_name
        # 设计变量个数
        self.m_NumDesignVars = num_design_vars
        # 上下界
        self.m_UpBounds = up_bounds
        self.m_LowBounds = low_bounds
        # 最大迭代次数
        self.m_MaxEvaluations = max_evaluations
        # 初值
        self.m_InitialValue = initial_value if initial_value is not None else []
        # 中间文件目录
        self.m_FilePath = file_path + "\\DesignVar_Obj.txt"

    def objective(self, design_vars):
        """
        计算目标函数的值。
        先将设计变量的值写入指定文件，等待另一个进程将目标函数的值追加到文件末尾，然后读取该值。

        :param design_vars: 设计变量的值
        :return: 目标函数的值
        """        
        # 写入设计变量（带排他锁）
        while True:
            try:
                with open(self.m_FilePath, 'w') as f:
                    portalocker.lock(f, portalocker.LOCK_EX)  # 获取排他锁
                    line = ' '.join(str(var) for var in design_vars)
                    f.write(line + '\n' + 'DVWrited' )
                    portalocker.unlock(f)  # 立即释放锁
                    break
            except :
                pass
                #time.sleep(0.1)  # 如果锁被其他进程占用，等待一段时间后重试

        # 等待C++进程完成计算
        while True:
            try:
                with open(self.m_FilePath, 'r') as f:
                    lines = f.readlines()

                    # 如果文件为空，则继续尝试
                    if not lines:
                        pass
                    else:
                        # 获取最后一行，去掉换行符
                        last_line = lines[-1].strip()
                        # 如果最后一行是 ObjWrited ，读取倒数第二行
                        if last_line == "ObjWrited" and len(lines) > 1:
                            return float(lines[-2].strip())
                        
            except Exception as e:
                pass
                #print(f"打开文件失败: {e}，正在重试...")

       
    def StartOptimization(self):
        """ 启动优化流程 """
        if self.m_AlgorithmName == "RBFopt":
            # 配置RBFopt参数
            settings = RbfoptSettings(
                max_evaluations=self.m_MaxEvaluations,
            )
            
            # 创建黑盒对象
            bb = RbfoptUserBlackBox(
                dimension=self.m_NumDesignVars,
                var_lower=np.array(self.m_LowBounds),
                var_upper=np.array(self.m_UpBounds),
                var_type=['R']*self.m_NumDesignVars,  # 连续变量
                obj_funct=lambda x: self.objective(x)
            )
            
            # 执行优化
            alg = RbfoptAlgorithm(settings, bb)
             # 执行优化并获取完整结果
            result = alg.optimize()
            return result[1], result[0], alg.itercount  # 新增迭代次数
            
        elif self.m_AlgorithmName == "CMAES":
            # 生成初始解（优先使用用户提供的初值）
            x0 = self.m_InitialValue if self.m_InitialValue else [
                (u + l)/2 for u, l in zip(self.m_UpBounds, self.m_LowBounds)
            ]
            
            # 设置变量边界
            # 修正边界设置方式
            bounds = [self.m_LowBounds, self.m_UpBounds]  # 改为二维列表格式
            
            # 执行CMA-ES优化
            result = cma.fmin(
                self.objective, 
                x0, 
                0.5,
                options={
                    'maxfevals': self.m_MaxEvaluations,
                    'bounds': bounds,
                    'verb_disp': 1 ,# 显示基本信息
                    'verbose': -9  # 新增参数：-9表示完全静默
                },
                restarts=2
            )
            return result[0], result[1], result[2]  # result[2]包含评估次数
        
        else:
            raise ValueError(f"不支持的优化算法: {self.m_AlgorithmName}")
        

def create_optimizer_from_config(config_path):
    """
        从配置文件初始化优化器
        配置文件格式示例：
        algorithm_name: RBFopt
        num_design_vars: 2
        up_bounds: 3 3
        low_bounds: 0 0
        max_evaluations: 50
        file_path: D:\VS_Repos\PKPMOpti\PythonApplication1\Solvers
        initial_value: 1 1
        """
    config = {}
    with open(config_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # 处理列表型参数
                if ' ' in value:
                    try:
                        config[key] = [float(x) if '.' in x else int(x) for x in value.split()]
                    except ValueError:
                        config[key] = value.split()
                else:
                    try:
                        config[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        config[key] = value

    return Opti_BlackBox(
        algorithm_name=config['algorithm_name'],
        num_design_vars=config['num_design_vars'],
        up_bounds=config['up_bounds'],
        low_bounds=config['low_bounds'],
        max_evaluations=config['max_evaluations'],
        file_path=config['file_path'],
        initial_value=config.get('initial_value')
    )


def monitor_parent_process(parent_pid):
    """ 监测父进程是否存活，若父进程结束，则子进程也退出 """
    while True:
        if not psutil.pid_exists(parent_pid):
            print(f"父进程 {parent_pid} 已终止，子进程退出。")
            os._exit(0)  # 立即终止子进程
        time.sleep(1)  # 休眠一段时间，避免CPU占用过高


if __name__ == "__main__":
    
    import sys
    if len(sys.argv) < 3:  # 修改参数检查
        print("用法: python main.py <配置文件路径> <父进程ID>")
        sys.exit(1)
        
    # 启动监控线程
    parent_pid = int(sys.argv[2])
    monitor_thread = threading.Thread(
        target=monitor_parent_process,
        args=(parent_pid,),
        daemon=True
    )
    monitor_thread.start()

    # 添加_internal目录到搜索路径
    exe_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    internal_path = os.path.join(exe_dir, '_internal')
    os.environ["PATH"] += os.pathsep + internal_path
    
    config_path = sys.argv[1]


    #config_path = r"D:\项目\宁波中石化优化后-PK模型\宁波中石化-PK模型\Opt\PKPMOpti.Config"
    opti = create_optimizer_from_config(config_path)
    BestDV, BestValue, Iteration = opti.StartOptimization()

    while True:
            try:                
                with open(opti.m_FilePath, 'w') as f:
                    portalocker.lock(f, portalocker.LOCK_EX)  # 获取排他锁
                    #判断bestdv是否为数字
                    if isinstance(BestDV, int) or isinstance(BestDV, float):
                        line = str(BestDV) 
                        f.write(line) #f.write(str(BestValue)) #f.write(str(Iteration))
                    else:
                        line = ' '.join(str(var) for var in BestDV)
                        f.write(line) #f.write(str(BestValue)) #f.write(str(Iteration))
                    f.write('\n' + 'OptiFinish' )
                    portalocker.unlock(f)  # 立即释放锁
                    break
            except :
                pass


