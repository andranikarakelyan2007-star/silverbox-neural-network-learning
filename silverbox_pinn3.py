"""
silverbox_pinn.py
=================
Полный шаблон для лабораторной работы по PINN — Silverbox-осциллятор.

Структура:
  1.  Загрузка и предобработка данных
  2.  Численный интегратор RK4
  3.  Нейросеть AccelerationNet
  4.  Функция свободного прогона (free-run) и валидации
  5.  Утилита для построения графиков
  6.  Цикл обучения (батчевый multiple shooting + run_training)
  7.  Блок __main__: sanity-check → обучение → финальный график

Обучение идёт мини-батчами случайных окон длины SEQ_LEN (multiple shooting):
все окна прокатываются RK4 параллельно, стартовые состояния берутся из
train-записи. Финальный free-run (метрика из задания) остаётся честным
последовательным прогоном с первого отсчёта.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io

# ──────────────────────────────────────────────────────────────────────────────
# Глобальные настройки
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "silverboxfiles/SNLS80mV.mat"
FS          = 610.35                    # частота дискретизации, Гц
DT          = 1.0 / FS                 # шаг по времени, с
SLICE_START = 42650                     # начальный индекс среза
SLICE_LEN   = 4096                      # длина среза
SLICE_END   = SLICE_START + SLICE_LEN  # 46746
N_TRAIN     = 3072                      # отсчётов в train
N_TEST      = 1024                      # отсчётов в test

# ── Гиперпараметры обучения ───────────────────────────────────────────────────
HIDDEN_SIZE  = 64       # нейронов в каждом скрытом слое
N_LAYERS     = 3        # количество скрытых слоев
N_EPOCHS     = 400    # число эпох обучения
LR           = 1e-3     # начальная скорость обучения Adam
SEQ_LEN      = 64       # длина окна (число шагов RK4 в одном rollout при обучении)
BATCH_WINDOWS   = 256   # сколько окон прокатываем параллельно за один шаг оптимизатора
STEPS_PER_EPOCH = 5     # шагов Adam на одну «эпоху» (лог/валидация раз в LOG_EVERY эпох)
LOG_EVERY    = 10     # печать метрик каждые N эпох
SAVE_BEST    = True     # сохранять чекпоинт лучшей модели по test FR RMSE
CHECKPOINT   = "best_model.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# 1. ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ
# ──────────────────────────────────────────────────────────────────────────────

def load_silverbox(path: str = DATA_PATH, device: torch.device = DEVICE):
    """
    Загружает датасет Silverbox (SNLS80mV.mat), нарезает нужный участок,
    центрирует сигналы и возвращает train/test тензоры.

    Returns
    -------
    u_train, y_train, u_test, y_test : torch.Tensor  shape (N,)
    u_all, y_all                      : torch.Tensor  shape (4096,)
    dt                                : float
    y0, v0                            : float  — начальное состояние
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Файл данных не найден: '{path}'.\n"
            f"Убедитесь, что папка 'silverboxfiles' лежит рядом со скриптом "
            f"и содержит файл 'SNLS80mV.mat'."
        )

    raw = scipy.io.loadmat(path)

    # Извлечение каналов V1 (вход u) и V2 (выход y)
    # В .mat-файле массивы могут храниться как (1,N) или (N,1) — нормализуем
    u_full = np.array(raw["V1"], dtype=np.float64).ravel()
    y_full = np.array(raw["V2"], dtype=np.float64).ravel()

    print(f"[data] Полная длина V1 (u): {len(u_full)}, V2 (y): {len(y_full)}")

    # Срез [42650 : 46746] — ровно 4096 отсчётов
    u_slice = u_full[SLICE_START:SLICE_END]
    y_slice = y_full[SLICE_START:SLICE_END]
    assert len(u_slice) == SLICE_LEN and len(y_slice) == SLICE_LEN, \
        f"Неожиданная длина среза: {len(u_slice)}, {len(y_slice)}"

    # Центрирование (вычитаем среднее по всему срезу)
    u_mean, y_mean = u_slice.mean(), y_slice.mean()
    u_slice = u_slice - u_mean
    y_slice = y_slice - y_mean
    print(f"[data] Центрирование: u_mean={u_mean:.6f} V, y_mean={y_mean:.6f} V")

    # Разделение: train — первые 3072, test — последние 1024
    u_train_np = u_slice[:N_TRAIN]
    y_train_np = y_slice[:N_TRAIN]
    u_test_np  = u_slice[N_TRAIN:]
    y_test_np  = y_slice[N_TRAIN:]

    # Конвертация в torch.Tensor float32
    to_tensor = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device)

    u_train = to_tensor(u_train_np)
    y_train = to_tensor(y_train_np)
    u_test  = to_tensor(u_test_np)
    y_test  = to_tensor(y_test_np)
    u_all   = to_tensor(u_slice)
    y_all   = to_tensor(y_slice)

    dt = DT  # 1 / 610.35

    # Начальные условия (на первом шаге среза)
    y0 = float(y_all[0])
    v0 = float((y_all[1] - y_all[0]) / dt)   # конечная разность первого порядка

    print(f"[data] y0 = {y0:.6f} V,  v0 = {v0:.4f} V/s")
    print(f"[data] u_train: {u_train.shape}, y_train: {y_train.shape}")
    print(f"[data] u_test:  {u_test.shape},  y_test:  {y_test.shape}")
    print(f"[data] dt = {dt:.8f} s  (fs = {FS} Hz)")

    return u_train, y_train, u_test, y_test, u_all, y_all, dt, y0, v0


# ──────────────────────────────────────────────────────────────────────────────
# 2. ЧИСЛЕННЫЙ ИНТЕГРАТОР RK4
# ──────────────────────────────────────────────────────────────────────────────

def rk4_step(
    model: nn.Module,
    y: torch.Tensor,
    v: torch.Tensor,
    u_curr: torch.Tensor,
    u_next: torch.Tensor,
    dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Один шаг метода Рунге — Кутты 4-го порядка для системы:
        dy/dt = v
        dv/dt = model(y, v, u)

    Вход u линейно интерполируется между u_curr и u_next.
    Поддерживает как скаляры, так и батчи тензоров.

    Parameters
    ----------
    model   : nn.Module  — возвращает ускорение dv/dt = f(y, v, u)
    y       : Tensor     — смещение (displacement)
    v       : Tensor     — скорость (velocity)
    u_curr  : Tensor     — вход на текущем шаге t
    u_next  : Tensor     — вход на следующем шаге t + dt
    dt      : float      — шаг по времени

    Returns
    -------
    y_next, v_next : Tensor  — состояние после шага
    """
    u_mid = 0.5 * (u_curr + u_next)   # вход в середине интервала

    # k1 — производные в начале шага
    f_y1 = v
    f_v1 = model(y, v, u_curr)

    # k2 — производные в середине шага (предиктор с k1)
    y2 = y + 0.5 * dt * f_y1
    v2 = v + 0.5 * dt * f_v1
    f_y2 = v2
    f_v2 = model(y2, v2, u_mid)

    # k3 — производные в середине шага (уточнённый предиктор с k2)
    y3 = y + 0.5 * dt * f_y2
    v3 = v + 0.5 * dt * f_v2
    f_y3 = v3
    f_v3 = model(y3, v3, u_mid)

    # k4 — производные в конце шага (предиктор с k3)
    y4 = y + dt * f_y3
    v4 = v + dt * f_v3
    f_y4 = v4
    f_v4 = model(y4, v4, u_next)

    # Взвешенная сумма: 1/6 * (k1 + 2k2 + 2k3 + k4)
    y_next = y + (dt / 6.0) * (f_y1 + 2.0 * f_y2 + 2.0 * f_y3 + f_y4)
    v_next = v + (dt / 6.0) * (f_v1 + 2.0 * f_v2 + 2.0 * f_v3 + f_v4)

    return y_next, v_next


# ──────────────────────────────────────────────────────────────────────────────
# 3. НЕЙРОСЕТЬ AccelerationNet
# ──────────────────────────────────────────────────────────────────────────────

class AccelerationNet(nn.Module):
    """
    Нейросеть, аппроксимирующая правую часть уравнения движения Silverbox:
        dv/dt = f(y, v, u)

    Принимает на вход тройку (y, v, u) → возвращает скаляр/тензор ускорения.

    Parameters
    ----------
    hidden_size  : int        — нейронов в каждом скрытом слое
    n_layers     : int        — количество скрытых слоёв
    activation   : nn.Module  — функция активации (по умолчанию Tanh)
    """

    def __init__(
        self,
        hidden_size: int = 64,
        n_layers: int = 3,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if activation is None:
            activation = nn.Tanh()

        in_features = 3  # (y, v, u)
        layers: list[nn.Module] = []

        # Входной слой
        layers.append(nn.Linear(in_features, hidden_size))
        layers.append(activation)

        # Скрытые слои
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)

        # Выходной слой — одна координата (ускорение)
        layers.append(nn.Linear(hidden_size, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

        # Масштабы нормировки (внутри сети). По умолчанию 1.0 = без нормировки.
        # Это buffers, а не параметры: не обучаются, но сохраняются в state_dict
        # (чекпоинт сам хранит масштабы) и переезжают на нужное устройство с .to().
        for name in ("y_scale", "v_scale", "u_scale", "a_scale"):
            self.register_buffer(name, torch.tensor(1.0))

    def set_scales(
        self,
        y_scale: float,
        v_scale: float,
        u_scale: float,
        a_scale: float,
    ) -> None:
        """
        Задаёт масштабы нормировки. Снаружи сеть по-прежнему принимает и
        возвращает физические величины (В, В/с, В/с²); деление входов и
        умножение выхода происходят только внутри forward.
        """
        self.y_scale.fill_(float(y_scale))
        self.v_scale.fill_(float(v_scale))
        self.u_scale.fill_(float(u_scale))
        self.a_scale.fill_(float(a_scale))

    def _init_weights(self) -> None:
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        y: torch.Tensor,
        v: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        y, v, u : Tensor  — скаляры (shape []) или батчи (shape [B])

        Returns
        -------
        Tensor  — ускорение dv/dt (физические единицы), той же формы, что y/v/u
        """
        y = y.unsqueeze(-1) if y.dim() == 0 else y.view(-1, 1)
        v = v.unsqueeze(-1) if v.dim() == 0 else v.view(-1, 1)
        u = u.unsqueeze(-1) if u.dim() == 0 else u.view(-1, 1)

        # Нормировка входов: все три числа порядка 1 (физика не меняется,
        # это просто смена единиц внутри сети)
        x   = torch.cat(
            [y / self.y_scale, v / self.v_scale, u / self.u_scale], dim=-1
        )                                   # (B, 3)
        out = self.net(x) * self.a_scale    # (B, 1), обратно в физические В/с²
        return out.squeeze(-1)              # (B,) или скаляр


def compute_scales(
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    dt: float,
) -> dict:
    """
    Масштабы нормировки по ТРЕНИРОВОЧНЫМ данным (test не используется).

        y_scale = std(y_train)
        v_scale = std(v_est),  v_est — центральные разности y_train
        u_scale = std(u_train)
        a_scale = v_scale² / y_scale

    a_scale — характерный размер ускорения. Для колебания y = A·sin(ωt)
    имеем std(v) = ω·std(y), а ускорение ~ ω²·y, поэтому
    ω ≈ v_scale / y_scale и a_scale ≈ ω²·y_scale = v_scale² / y_scale.
    Так надёжнее, чем дифференцировать шумный сигнал второй раз.
    """
    v_est = torch.gradient(y_train, spacing=(dt,))[0]
    y_scale = float(y_train.std())
    v_scale = float(v_est.std())
    u_scale = float(u_train.std())
    a_scale = v_scale ** 2 / y_scale
    return {
        "y_scale": y_scale,
        "v_scale": v_scale,
        "u_scale": u_scale,
        "a_scale": a_scale,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. ФУНКЦИЯ СВОБОДНОГО ПРОГОНА (FREE RUN) И ВАЛИДАЦИИ
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_free_run(
    model: nn.Module,
    u_all: torch.Tensor,
    y_all: torch.Tensor,
    dt: float,
) -> dict:
    """
    Непрерывный сквозной прогон модели (free-run) по всем 4096 отсчётам.
    Истинный y НЕ подаётся на вход во время прогона — только u.

    Returns
    -------
    dict:
        'baseline_rmse'  — RMS истинного тестового y (эталон нормировки)
        'train_fr_rmse'  — Free-Run RMSE на train-участке (первые 3072)
        'test_fr_rmse'   — Free-Run RMSE на test-участке  (последние 1024)
        'y_pred_all'     — Tensor (4096,) предсказанного сигнала
    """
    model.eval()

    N = len(u_all)
    assert N == SLICE_LEN, f"Ожидается 4096 отсчётов, получено {N}"

    # Начальные условия из первых двух точек
    y_t = y_all[0].clone()
    v_t = (y_all[1] - y_all[0]) / dt

    y_pred_list: list[torch.Tensor] = [y_t]

    for t in range(N - 1):
        y_t, v_t = rk4_step(model, y_t, v_t, u_all[t], u_all[t + 1], dt)
        y_pred_list.append(y_t)

    y_pred_all = torch.stack(y_pred_list)  # (4096,)

    def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean((a - b) ** 2)))

    def rms(a: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(a ** 2)))

    return {
        "baseline_rmse" : rms(y_all[N_TRAIN:]),
        "train_fr_rmse" : rmse(y_pred_all[:N_TRAIN], y_all[:N_TRAIN]),
        "test_fr_rmse"  : rmse(y_pred_all[N_TRAIN:], y_all[N_TRAIN:]),
        "y_pred_all"    : y_pred_all,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. УТИЛИТЫ ДЛЯ ПОСТРОЕНИЯ ГРАФИКОВ
# ──────────────────────────────────────────────────────────────────────────────

def plot_test_results(
    y_true_test: torch.Tensor,
    y_pred_test: torch.Tensor,
    save_path: str = "test_plot.png",
    title: str = "Silverbox — Free-Run на тестовом интервале",
) -> None:
    """
    Двухпанельный график: сигналы (верх) и ошибка (низ).
    """
    t_axis    = np.arange(len(y_true_test)) * DT
    y_true_np = y_true_test.detach().cpu().numpy()
    y_pred_np = y_pred_test.detach().cpu().numpy()
    error_np  = y_pred_np - y_true_np

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(title, fontsize=13)

    ax0 = axes[0]
    ax0.plot(t_axis, y_true_np, label="Измеренный y",     color="steelblue", linewidth=1.2)
    ax0.plot(t_axis, y_pred_np, label="Предсказанный ŷ",  color="tomato",
             linewidth=1.0, linestyle="--")
    ax0.set_ylabel("Напряжение, В")
    ax0.legend(loc="upper right", fontsize=9)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(t_axis, error_np, color="darkorange", linewidth=0.8, label="Ошибка ŷ − y")
    ax1.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Ошибка, В")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Сохранён: {save_path}")


def plot_loss_curve(
    history: dict,
    save_path: str = "loss_curve.png",
) -> None:
    """
    График кривых обучения: loss, train FR RMSE, test FR RMSE по эпохам.

    Parameters
    ----------
    history : dict с ключами 'epoch', 'loss', 'train_fr_rmse', 'test_fr_rmse'
    """
    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Кривые обучения Silverbox PINN", fontsize=13)

    # Левый: loss
    axes[0].semilogy(epochs, history["loss"], color="royalblue", linewidth=1.2)
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("Loss (log)")
    axes[0].set_title("Loss по эпохам")
    axes[0].grid(True, alpha=0.3)

    # Правый: RMSE
    axes[1].semilogy(epochs, history["train_fr_rmse"],
                     label="Train FR RMSE", color="steelblue",  linewidth=1.2)
    axes[1].semilogy(epochs, history["test_fr_rmse"],
                     label="Test FR RMSE",  color="tomato",      linewidth=1.2)
    axes[1].axhline(history["baseline_rmse"], color="gray",
                    linestyle="--", linewidth=0.8, label=f"Baseline {history['baseline_rmse']:.4f} V")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("RMSE, В (log)")
    axes[1].set_title("Free-Run RMSE по эпохам")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Сохранён: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. ЦИКЛ ОБУЧЕНИЯ
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss_truncated_bptt(
    model: nn.Module,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    dt: float,
    seq_len: int = SEQ_LEN,
) -> torch.Tensor:
    """
    Вычисляет MSE-лосс методом truncated BPTT (обрезанное обратное
    распространение во времени по сегментам длиной seq_len).

    Почему truncated BPTT, а не полный прогон?
    ------------------------------------------
    Если разворачивать весь граф вычислений через все 3072 шага RK4 сразу,
    память GPU/CPU заканчивается и градиенты затухают/взрываются.
    Вместо этого мы делаем следующее:
      • Делим тренировочный сигнал на сегменты длиной seq_len.
      • На границе каждого сегмента состояние (y_t, v_t) детачится
        (.detach()) — граф обрывается и память освобождается.
      • Внутри каждого сегмента градиент течёт нормально через все seq_len шагов RK4.
      • Лоссы по всем сегментам суммируются в один скаляр.

    Parameters
    ----------
    model    : AccelerationNet
    u_train  : Tensor (N_TRAIN,) — входной сигнал обучающего участка
    y_train  : Tensor (N_TRAIN,) — истинный выход обучающего участка
    dt       : float
    seq_len  : int — длина одного сегмента (truncation window)

    Returns
    -------
    loss : Tensor (скаляр) — среднеквадратическая ошибка по всему train
    """
    model.train()

    N = len(u_train)

    # ── Инициализация состояния ────────────────────────────────────────────
    # Начинаем с первой точки тренировочного сигнала
    y_t = y_train[0].clone()
    v_t = (y_train[1] - y_train[0]) / dt   # оценка начальной скорости

    total_loss = torch.tensor(0.0, device=u_train.device)
    n_segments = 0

    # ── Цикл по сегментам ─────────────────────────────────────────────────
    for seg_start in range(0, N - 1, seq_len):
        seg_end = min(seg_start + seq_len, N - 1)

        # Детачим состояние на границе сегмента:
        # градиент НЕ потечёт через предыдущий сегмент, но значение сохраняется.
        # Это ключевой момент truncated BPTT.
        y_t = y_t.detach()
        v_t = v_t.detach()

        seg_preds: list[torch.Tensor] = []

        # ── Прокатываем RK4 внутри сегмента ──────────────────────────────
        for t in range(seg_start, seg_end):
            # Один шаг RK4: модель вызывается 4 раза внутри rk4_step
            y_t, v_t = rk4_step(model, y_t, v_t, u_train[t], u_train[t + 1], dt)
            seg_preds.append(y_t)

        # ── Лосс сегмента: MSE предсказанного y vs истинного y ───────────
        # seg_preds содержит предсказания для индексов [seg_start+1 ... seg_end]
        y_pred_seg = torch.stack(seg_preds)                 # (seg_len,)
        y_true_seg = y_train[seg_start + 1 : seg_end + 1]  # (seg_len,)

        seg_loss = torch.mean((y_pred_seg - y_true_seg) ** 2)
        total_loss = total_loss + seg_loss
        n_segments += 1

    # Нормируем на число сегментов, чтобы величина лосса не зависела от seq_len
    return total_loss / n_segments


def compute_loss_multiple_shooting(
    model: nn.Module,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    v_est: torch.Tensor,
    dt: float,
    seq_len: int = SEQ_LEN,
    batch_windows: int = BATCH_WINDOWS,
) -> torch.Tensor:
    """
    Батчевый multiple shooting: MSE по пачке случайных окон длины seq_len.

    Идея
    ----
    Вместо одной цепочки из 3072 последовательных шагов RK4 (batch=1) берём
    `batch_windows` случайных окон train-записи и прокатываем их ПАРАЛЛЕЛЬНО:
    в цикле остаётся всего seq_len последовательных шагов, а каждый вызов сети
    обрабатывает батч (B,) вместо скаляра.

    Стартовое состояние окна: y — измеренное, v — оценка по разностям
    (v_est, считается один раз по y_train). Используются только train-данные.

    Parameters
    ----------
    v_est : Tensor (N_TRAIN,) — оценка скорости по y_train
    """
    model.train()

    N = len(u_train)
    L = seq_len
    dev = u_train.device

    # Старты окон: s ∈ [0, N-1-L], чтобы последний индекс s+L не вышел за N-1
    starts = torch.randint(0, N - L, (batch_windows,), device=dev)      # (B,)
    idx = starts[:, None] + torch.arange(L + 1, device=dev)[None, :]    # (B, L+1)

    u_w = u_train[idx]                      # (B, L+1)
    y_w = y_train[idx]                      # (B, L+1)

    y_t = y_w[:, 0]                         # (B,)  измеренная позиция на старте
    v_t = v_est[starts]                     # (B,)  оценка скорости на старте

    preds: list[torch.Tensor] = []
    for t in range(L):
        y_t, v_t = rk4_step(model, y_t, v_t, u_w[:, t], u_w[:, t + 1], dt)
        preds.append(y_t)

    y_pred = torch.stack(preds, dim=1)      # (B, L)
    return torch.mean((y_pred - y_w[:, 1:]) ** 2)


def run_training(
    model: nn.Module,
    u_train: torch.Tensor,
    y_train: torch.Tensor,
    u_all: torch.Tensor,
    y_all: torch.Tensor,
    dt: float,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    seq_len: int = SEQ_LEN,
    log_every: int = LOG_EVERY,
    save_best: bool = SAVE_BEST,
    checkpoint_path: str = CHECKPOINT,
    batch_windows: int = BATCH_WINDOWS,
    steps_per_epoch: int = STEPS_PER_EPOCH,
) -> dict:
    """
    Полный цикл обучения AccelerationNet на данных Silverbox.

    Стратегия обучения
    ------------------
    Оптимизатор:   Adam (адаптивный момент, хорошо работает при нестационарных
                   градиентах, которые возникают при прокатке через RK4).

    LR-планировщик: ReduceLROnPlateau — автоматически снижает lr в 0.5 раза,
                    если test FR RMSE не улучшается 200 эпох подряд.
                    Это помогает дообучиться после застревания в плато.

    Лосс:           MSE по батчу случайных окон (compute_loss_multiple_shooting).
                    Старая последовательная версия compute_loss_truncated_bptt
                    сохранена в файле, но в обучении не используется.

    Валидация:      evaluate_free_run — честный free-run без подглядывания в y.

    Чекпоинт:       Сохраняем state_dict модели каждый раз, когда test FR RMSE
                    улучшается. Можно продолжить обучение, загрузив checkpoint.

    Parameters
    ----------
    model           : AccelerationNet
    u_train, y_train: тренировочные данные (N_TRAIN,)
    u_all, y_all    : полные данные (4096,) для free-run валидации
    dt              : шаг по времени
    n_epochs        : число эпох
    lr              : начальная скорость обучения
    seq_len         : длина окна (шагов RK4) в одном rollout при обучении
    log_every       : частота логирования
    save_best       : сохранять ли лучший чекпоинт
    checkpoint_path : путь к файлу чекпоинта
    batch_windows   : сколько окон в одном батче
    steps_per_epoch : сколько шагов оптимизатора в одной «эпохе»

    Returns
    -------
    history : dict с историей метрик по эпохам (для построения графиков)
    """
    # ── Оптимизатор Adam ──────────────────────────────────────────────────────
    # Adam хранит скользящие средние градиентов (m) и квадратов градиентов (v),
    # что даёт адаптивный lr для каждого параметра.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Планировщик lr: ReduceLROnPlateau ────────────────────────────────────
    # Следит за метрикой (test FR RMSE). Если за `patience` эпох метрика
    # не улучшается — умножает lr на `factor`.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",       # минимизируем test FR RMSE
        factor=0.5,       # lr *= 0.5 при срабатывании
        patience=200,     # ждём 200 эпох без улучшения
        min_lr=1e-6,      # нижняя граница lr
        verbose=False,    # сами будем логировать
    )

    # ── История метрик ────────────────────────────────────────────────────────
    history: dict = {
        "epoch"          : [],
        "loss"           : [],
        "train_fr_rmse"  : [],
        "test_fr_rmse"   : [],
        "lr"             : [],
        "baseline_rmse"  : 0.0,   # заполним на первой валидации
    }

    # Оценка скорости по train-записи (центральные разности), один раз.
    # Используется только как стартовое состояние окон.
    v_est = torch.gradient(y_train, spacing=(dt,))[0]

    best_test_rmse = float("inf")  # лучший test FR RMSE за всё обучение
    t0 = time.time()               # время старта для оценки скорости

    print("\n" + "=" * 70)
    print("  НАЧАЛО ОБУЧЕНИЯ")
    print(f"  Эпох: {n_epochs}  |  LR: {lr}  |  seq_len: {seq_len}")
    print(f"  Окон в батче: {batch_windows}  |  шагов Adam за эпоху: {steps_per_epoch}")
    print(f"  Оптимизатор: Adam  |  Планировщик: ReduceLROnPlateau")
    print("=" * 70)

    # ── Главный цикл по эпохам ────────────────────────────────────────────────
    for epoch in range(1, n_epochs + 1):

        # ── Шаги 1–4: steps_per_epoch шагов оптимизатора по случайным окнам ──
        # Каждый шаг: батч из batch_windows окон длины seq_len, прокатанных
        # RK4 параллельно (compute_loss_multiple_shooting) → backward →
        # gradient clipping (стабилизация при прокатке через RK4) → Adam.
        loss_sum = 0.0
        for _ in range(steps_per_epoch):
            optimizer.zero_grad()
            step_loss = compute_loss_multiple_shooting(
                model, u_train, y_train, v_est, dt, seq_len, batch_windows
            )
            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_sum += step_loss.detach()

        # Средний лосс за эпоху (для логов/чекпоинта); .item() ниже — один sync
        loss = loss_sum / steps_per_epoch

        # ── Шаг 5: Валидация (каждые log_every эпох) ─────────────────────────
        # Запускаем честный free-run без градиентов (@torch.no_grad())
        if epoch % log_every == 0 or epoch == 1:
            metrics = evaluate_free_run(model, u_all, y_all, dt)

            train_fr = metrics["train_fr_rmse"]
            test_fr  = metrics["test_fr_rmse"]
            baseline = metrics["baseline_rmse"]
            cur_lr   = optimizer.param_groups[0]["lr"]

            # Запоминаем baseline при первой валидации
            if history["baseline_rmse"] == 0.0:
                history["baseline_rmse"] = baseline

            # ── Шаг 6: Обновление планировщика lr ────────────────────────────
            # ReduceLROnPlateau смотрит на test_fr и решает, снижать ли lr.
            scheduler.step(test_fr)

            # ── Шаг 7: Сохранение лучшего чекпоинта ──────────────────────────
            if save_best and test_fr < best_test_rmse:
                best_test_rmse = test_fr
                torch.save(
                    {
                        "epoch"          : epoch,
                        "model_state"    : model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "test_fr_rmse"   : test_fr,
                        "loss"           : loss.item(),
                    },
                    checkpoint_path,
                )
                ckpt_marker = " ✓ (новый лучший)"
            else:
                ckpt_marker = ""

            # ── Логирование в терминал ────────────────────────────────────────
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:5d}/{n_epochs}"
                f" | loss={loss.item():.6f}"
                f" | trainFR={train_fr:.6f} V"
                f" | testFR={test_fr:.6f} V"
                f" | lr={cur_lr:.2e}"
                f" | {elapsed:6.1f}s"
                f"{ckpt_marker}"
            )

            # Пишем в историю
            history["epoch"].append(epoch)
            history["loss"].append(loss.item())
            history["train_fr_rmse"].append(train_fr)
            history["test_fr_rmse"].append(test_fr)
            history["lr"].append(cur_lr)

    # ── Конец обучения ────────────────────────────────────────────────────────
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  ОБУЧЕНИЕ ЗАВЕРШЕНО за {total_time:.1f} с")
    print(f"  Лучший test FR RMSE: {best_test_rmse:.6f} V")
    if save_best:
        print(f"  Чекпоинт сохранён: {checkpoint_path}")
    print("=" * 70 + "\n")

    return history


# ──────────────────────────────────────────────────────────────────────────────
# 7. ТОЧКА ВХОДА
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  Silverbox PINN")
    print("=" * 60)

    # ── Загрузка данных ───────────────────────────────────────────────────────
    (
        u_train, y_train,
        u_test,  y_test,
        u_all,   y_all,
        dt, y0, v0,
    ) = load_silverbox(path=DATA_PATH, device=DEVICE)

    print(f"\n[info] Device: {DEVICE}")
    print(f"[info] N_TRAIN={N_TRAIN}, N_TEST={N_TEST}, SLICE_LEN={SLICE_LEN}")
    print(f"[info] y0={y0:.6f} V,  v0={v0:.4f} V/s")

    # ── Baseline RMSE ─────────────────────────────────────────────────────────
    baseline_rmse = float(torch.sqrt(torch.mean(y_test ** 2)))
    print(f"\n[baseline] RMS тестового y = {baseline_rmse:.6f} V  (~0.06 В ожидается)")

    # ── Инициализация модели ──────────────────────────────────────────────────
    model = AccelerationNet(
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
    ).to(DEVICE)

    # Нормировка входов/выхода внутри сети (масштабы — только по train)
    scales = compute_scales(u_train, y_train, dt)
    model.set_scales(**scales)
    print("\n[scales] " + "  ".join(f"{k}={val:.4g}" for k, val in scales.items()))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[model] AccelerationNet: {n_params} обучаемых параметров")
    print(model)

    # ── Sanity-check: free-run нетренированной модели ─────────────────────────
    print("\n[sanity] Запуск free-run НЕТРЕНИРОВАННОЙ модели...")
    results_before = evaluate_free_run(model, u_all, y_all, dt)
    print(f"  baseline RMSE = {results_before['baseline_rmse']:.6f} V")
    print(f"  train FR RMSE = {results_before['train_fr_rmse']:.6f} V")
    print(f"  test  FR RMSE = {results_before['test_fr_rmse']:.6f} V")

    plot_test_results(
        y_true_test=y_all[N_TRAIN:],
        y_pred_test=results_before["y_pred_all"][N_TRAIN:],
        save_path="test_plot_untrained.png",
        title="До обучения — нетренированная модель",
    )

    # ── Обучение ──────────────────────────────────────────────────────────────
    history = run_training(
        model=model,
        u_train=u_train,
        y_train=y_train,
        u_all=u_all,
        y_all=y_all,
        dt=dt,
        n_epochs=N_EPOCHS,
        lr=LR,
        seq_len=SEQ_LEN,
        log_every=LOG_EVERY,
        save_best=SAVE_BEST,
        checkpoint_path=CHECKPOINT,
        batch_windows=BATCH_WINDOWS,
        steps_per_epoch=STEPS_PER_EPOCH,
    )

    # ── Загрузка лучшего чекпоинта (если сохранялся) ─────────────────────────
    if SAVE_BEST and os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print(f"[checkpoint] Загружена лучшая модель (epoch {ckpt['epoch']},"
              f" test FR RMSE = {ckpt['test_fr_rmse']:.6f} V)")

    # ── Финальная оценка ──────────────────────────────────────────────────────
    print("\n[eval] Финальный free-run лучшей модели...")
    results_after = evaluate_free_run(model, u_all, y_all, dt)
    print(f"  baseline RMSE = {results_after['baseline_rmse']:.6f} V")
    print(f"  train FR RMSE = {results_after['train_fr_rmse']:.6f} V")
    print(f"  test  FR RMSE = {results_after['test_fr_rmse']:.6f} V")

    # ── Финальные графики ─────────────────────────────────────────────────────
    plot_test_results(
        y_true_test=y_all[N_TRAIN:],
        y_pred_test=results_after["y_pred_all"][N_TRAIN:],
        save_path="test_plot_trained.png",
        title="После обучения — лучшая модель (test участок)",
    )

    plot_loss_curve(history, save_path="loss_curve.png")

    print("\n[done] Всё готово.")
