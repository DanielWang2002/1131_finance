import yfinance as yf
import numpy as np
import random

# 本範例將使用三支台股股票作為例子（台積電:2330.TW、聯電:2303.TW、鴻海:2317.TW）
# 時間範圍自行決定 (本例範例：2023-01-01 至 2023-12-31)
# 目標：透過基因演算法選擇最優投資標的 (只選一支股票)
# 適應度函數：使用平均日報酬率 - 風險 (日報酬率標準差) 作為Fitness
# GA 步驟：
#   1. 初始化族群（每個個體為一組基因，表示選擇哪一支股票）
#   2. 評估個體適應度
#   3. 選擇、交配、突變，生成新族群
#   4. 重複迭代，直到達到預定世代或收斂
#   因為本例只有三支股票，實務上GA意義不大(太簡單)，但此範例純為示意

# --------------------------
# 參數設定
# --------------------------
stocks = ["2330.TW", "2303.TW", "2317.TW"]  # 台積電、聯電、鴻海
start_date = "2023-01-01"
end_date = "2023-12-31"

pop_size = 10  # 族群大小
generations = 10  # 演化世代數
mutation_rate = 0.1

# --------------------------
# 抓取股票資料
# --------------------------
data = {}
for stock in stocks:
    df = yf.download(stock, start=start_date, end=end_date)
    data[stock] = df['Adj Close']

# 將資料對齊 (以防有缺值狀況)
prices = np.column_stack([data[s].values for s in stocks])
# 若有NaN，簡單以前值填補
where_nan = np.isnan(prices)
prices[where_nan] = np.nanmean(prices, axis=0)[np.where(where_nan)[1]]

# 計算每日報酬率
returns = (prices[1:] - prices[:-1]) / prices[:-1]
mean_returns = np.mean(returns, axis=0)
std_returns = np.std(returns, axis=0)


# --------------------------
# 適應度函數定義
# --------------------------
# 適應度：以 mean_return - std_return 為簡易衡量 (類似Sharpe概念，但無風險利率不考慮)
def fitness(chromosome):
    # chromosome為長度3的list, 例如 [1,0,0], 表示選擇stocks[0]
    # 我們確保只有一個基因為1
    idx = chromosome.index(1)
    m = mean_returns[idx]
    s = std_returns[idx]
    fit = m - s
    return fit


# --------------------------
# 基因演算法流程
# --------------------------


# 初始化族群
# 因為每個染色體為3長度的binary，但只允許一個1，其餘為0
def create_chromosome():
    # 隨機選1支
    ch = [0, 0, 0]
    ch[random.randint(0, 2)] = 1
    return ch


population = [create_chromosome() for _ in range(pop_size)]

for gen in range(generations):
    # 計算適應度
    fitness_values = [fitness(ch) for ch in population]

    # 印出本世代最佳
    best_fit = max(fitness_values)
    best_ch = population[fitness_values.index(best_fit)]
    print(f"Generation {gen+1}: Best Fitness = {best_fit:.6f}, Chromosome = {best_ch}")

    # 選擇(使用輪盤選擇或菁英選擇)
    # 本例使用簡單菁英加輪盤
    total_fitness = sum(fitness_values)
    # 若total_fitness小於0（理論上不會發生，若所有適應度都很差），加入位移
    if total_fitness <= 0:
        offset = abs(min(fitness_values))
        fitness_values = [f + offset + 0.0001 for f in fitness_values]
        total_fitness = sum(fitness_values)

    # 產生新族群
    new_population = []

    # 簡單菁英保存
    new_population.append(best_ch)

    # 輪盤選擇函式
    def roulette_wheel(fits):
        pick = random.random() * sum(fits)
        current = 0
        for i, f in enumerate(fits):
            current += f
            if current > pick:
                return i
        return len(fits) - 1  # Ensure a valid index is always returned

    while len(new_population) < pop_size:
        # 選擇兩個父母
        p1_idx = roulette_wheel(fitness_values)
        p2_idx = roulette_wheel(fitness_values)
        parent1 = population[p1_idx]
        parent2 = population[p2_idx]

        # 單點交配
        # 因為只能有一個基因=1，交配的實際意義有限，但仍示意
        cross_point = random.randint(1, 2)  # 在1或2點後切
        child1 = parent1[:cross_point] + parent2[cross_point:]
        child2 = parent2[:cross_point] + parent1[cross_point:]

        # 因為我們必須保證只有一支股票為1，若child不符則修正
        def fix_chromosome(ch):
            if sum(ch) == 1:
                return ch
            # 若不合格，隨機選一支股票為1，其它為0
            new_ch = [0, 0, 0]
            new_ch[random.randint(0, 2)] = 1
            return new_ch

        child1 = fix_chromosome(child1)
        child2 = fix_chromosome(child2)

        # 突變
        def mutate(ch):
            if random.random() < mutation_rate:
                # 突變則重新選一支股票
                m_ch = [0, 0, 0]
                m_ch[random.randint(0, 2)] = 1
                return m_ch
            return ch

        child1 = mutate(child1)
        child2 = mutate(child2)

        new_population.append(child1)
        if len(new_population) < pop_size:
            new_population.append(child2)

    population = new_population

# 最終結果
fitness_values = [fitness(ch) for ch in population]
best_fit = max(fitness_values)
best_ch = population[fitness_values.index(best_fit)]
best_stock = stocks[best_ch.index(1)]
final_idx = best_ch.index(1)
final_return = mean_returns[final_idx]
final_risk = std_returns[final_idx]

for i, stock in enumerate(stocks):
    print(
        f"{stock}: 平均回報率 = {mean_returns[i]:.6f}, 風險 = {std_returns[i]:.6f}, 適應度 = {mean_returns[i] - std_returns[i]:.6f}"
    )

print("\n===== 最終結果 =====")
print(f"股票選擇：{stocks}")
print(f"最佳投資組合選擇的股票: {best_stock}")
print(f"預期日平均回報率: {final_return:.6f}")
print(f"風險(報酬率標準差): {final_risk:.6f}")
print(f"適應度: {best_fit:.6f}")
