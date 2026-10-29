import torch
import time
### 3.1 Подготовка данных (5 баллов)
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами
A = torch.randn(64, 1024, 1024)
B = torch.randn(128, 512, 512)
C = torch.randn(256, 256, 256)
    
### 3.2 Функция измерения времени (5 баллов)
# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU
def measure_time_gpu(func, repeats):
    """Измерение времени выполнения функции на GPU"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # прогрев, как говорилось в лекции
    for _ in range(3):
        func()
    
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        func()
        
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeats # в миллисекундах

def measure_time_cpu(func, repeats):
    """Измерение времени выполнения функции на CPU"""
    # прогрев, как говорилось в лекции
    for _ in range(3):
        func()
    
    start_time = time.time()
    for _ in range(repeats):
        func()
        
    end_time = time.time()
    
    return (end_time - start_time) / repeats * 1000 # в миллисекундах

### 3.3 Сравнение операций (10 баллов)
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

def get_device():
    """Проверка доступности CUDA"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    print("CUDA не найдены")
    
    return torch.device("cpu")

def measure_cpu_time(func, *args, **kwargs):
    """Измерение времени выполнения функции на CPU."""
    # Прогрев (warmup) для честности измерений
    for _ in range(3):
        func(*args, **kwargs)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    for _ in range(5): # Усредняем по 5 итерациям
        func(*args, **kwargs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    return (end_time - start_time) / 5 * 1000 # в миллисекундах

def measure_gpu_time(func, *args, **kwargs):
    """Измерение времени выполнения функции на GPU с помощью событий CUDA."""
    device = args[0].device
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Прогрев
    for _ in range(3):
        func(*args, **kwargs)
    
    torch.cuda.synchronize()
    start_event.record()
    for _ in range(5):
        func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / 5 # в миллисекундах

def run_benchmark(*tensors : torch.Tensor):
    """3.1 - 3.3 Подготовка данных и сравнение операций."""
    device = get_device()
    operations = {
        "Матричное умножение": lambda A, B: torch.matmul(A, B),
        "Поэлементное сложение": lambda A, B: A + B,
        "Поэлементное умножение": lambda A, B: A * B,
        "Транспонирование": lambda A, B: A.transpose(1, 2),
        "Сумма элементов": lambda A, B: torch.sum(A)
    }
    
    print(f"{"Операция":<22} | {"Размер":<15} | {"CPU (мс)":<10} | {"GPU (мс)":<10} | {"Ускорение":<10}")
    for tensor in tensors:
        tensor_copy = tensor
        if device.type == "cuda":
            A_gpu = tensor.to(device)
            B_gpu = tensor_copy.to(device)
        else:
            A_gpu, B_gpu = None, None
            
        shape_str = f"{tensor.size(0)}x{tensor.size(1)}x{tensor.size(2)}"
        for op_name, op_func in operations.items():
            cpu_time = measure_cpu_time(op_func, tensor, tensor_copy)
            if device.type == "cuda":
                gpu_time = measure_gpu_time(op_func, A_gpu, B_gpu)
                speedup = cpu_time / gpu_time if gpu_time > 0 else "много"
                gpu_time_str = f"{gpu_time:.2f}"
                speedup_str = f"{speedup:.1f}x"
            else:
                gpu_time_str = "нет"
                speedup_str = "нет"
                
            print(f"{op_name:<22} | {shape_str:<15} | {cpu_time:<10.2f} | {gpu_time_str:<10} | {speedup_str:<10}")
            
run_benchmark(A, B, C)

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде
### Пример вывода:
# Операция          | CPU (мс) | GPU (мс) | Ускорение
# Матричное умножение|   150.2  |    12.3  |   12.2x
# Сложение          |    45.1  |     3.2  |   14.1x

### 3.4 Анализ результатов (5 баллов)
# Проанализируйте результаты:
# - Какие операции получают наибольшее ускорение на GPU?
# - Почему некоторые операции могут быть медленнее на GPU?
# - Как размер матриц влияет на ускорение?
# - Что происходит при передаче данных между CPU и GPU?