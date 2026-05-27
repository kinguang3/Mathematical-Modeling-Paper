import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.network import DeliveryNetwork
from model.capacity_model import CapacityPlanningModel
from model.scheduling_model import SchedulingModel
from algorithm.genetic_algorithm import GeneticAlgorithm
from algorithm.column_generation import ColumnGeneration
from algorithm.rolling_horizon import RollingHorizonScheduler
from utils.data_loader import DataLoader
from utils.metrics import MetricsCalculator, RobustnessAnalyzer
from simulation.simulator import DroneSimulator, WeatherSimulator
from simulation.visualization import NetworkVisualizer, ScheduleVisualizer, ResilienceVisualizer

def solve_problem_one(network, drone_capacity):
    print("=" * 50)
    print("问题一：网络流规划与最小机队配置")
    print("=" * 50)
    
    model = CapacityPlanningModel(network, drone_capacity)
    results = model.solve()
    
    print(f"求解状态: {results['status']}")
    print(f"最小无人机数量: {results['total_drones']}")
    print("各起降点航班分配:")
    for node, flights in results['flights'].items():
        print(f"  {node}: {flights} 架次")
    
    ga = GeneticAlgorithm(network, drone_capacity)
    ga_results = ga.solve()
    print(f"\n遗传算法结果: {ga_results['total_drones']} 架")
    
    # 保存结果到CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'capacity_optimal.csv'), 'w', encoding='utf-8') as f:
        f.write('起降点,航班数(架次)\n')
        for node, flights in results['flights'].items():
            f.write(f'{node},{flights}\n')
        f.write(f'总计,{results["total_drones"]}\n')
    
    print(f"\n问题一结果已保存到 data/results/capacity_optimal.csv")
    
    return results['total_drones']

def solve_problem_two(network, num_drones):
    print("\n" + "=" * 50)
    print("问题二：气象扰动下的实时调度")
    print("=" * 50)
    
    scheduler = SchedulingModel(network, drone_capacity=20)
    scheduler.generate_initial_schedule(num_drones)
    
    print(f"初始调度航班数: {len(scheduler.schedule)}")
    
    weather_sim = WeatherSimulator(network)
    events = weather_sim.generate_weather_events(60)
    
    if events:
        event = events[0]
        print(f"模拟气象扰动: 航段 {event['segment']} 关闭 {event['duration']} 分钟")
        
        affected_flights = [f for f in scheduler.schedule if (f['from'], f['to']) == event['segment']]
        print(f"受影响航班数: {len(affected_flights)}")
        
        rescheduled = scheduler.reschedule_after_disruption(event['segment'], affected_flights)
        print(f"重新调度完成，延误时间: {scheduler.calculate_total_delay()} 分钟")
    
    # 保存调度日志
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results')
    
    with open(os.path.join(results_dir, 'schedule_log.csv'), 'w', encoding='utf-8') as f:
        f.write('航班ID,起点,终点,时间,状态\n')
        for i, flight in enumerate(scheduler.schedule):
            status = '正常' if flight.get('status', 'normal') == 'normal' else '延误'
            f.write(f'{i+1},{flight["from"]},{flight["to"]},{flight.get("time", "N/A")},{status}\n')
    
    print(f"\n问题二结果已保存到 data/results/schedule_log.csv")
    
    return scheduler

def solve_problem_three(network):
    print("\n" + "=" * 50)
    print("问题三：关键节点识别与韧性评估")
    print("=" * 50)
    
    resilience = MetricsCalculator.calculate_network_resilience(network)
    print("各起降点失效后的韧性指标 Φ:")
    for node, phi in resilience.items():
        print(f"  {node}: {phi:.4f}")
    
    critical_nodes = MetricsCalculator.identify_critical_nodes(network, top_k=2)
    print(f"\n最关键的起降点: {[node for node, _ in critical_nodes]}")
    
    analyzer = RobustnessAnalyzer()
    strategy_result = analyzer.evaluate_reinforcement_strategy(network, [critical_nodes[0][0]])
    print("\n加固策略效果:")
    for node, improvement in strategy_result['improvement'].items():
        if improvement > 0:
            print(f"  {node}: 韧性提升 {improvement:.4f}")
    
    # 保存韧性评估结果
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results')
    
    with open(os.path.join(results_dir, 'resilience_results.csv'), 'w', encoding='utf-8') as f:
        f.write('起降点,韧性指标Φ,关键程度\n')
        sorted_nodes = sorted(resilience.items(), key=lambda x: x[1])
        for i, (node, phi) in enumerate(sorted_nodes):
            critical_level = '最关键' if i == 0 else '较关键' if i == 1 else '一般'
            f.write(f'{node},{phi:.4f},{critical_level}\n')
    
    print(f"\n问题三结果已保存到 data/results/resilience_results.csv")
    
    return resilience

def main():
    parser = argparse.ArgumentParser(description='城市低空即时配送网络优化')
    parser.add_argument('--problem', type=int, choices=[1, 2, 3, None], default=None,
                        help='选择要解决的问题 (1/2/3)，默认全部解决')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    args = parser.parse_args()
    
    N = 5
    flight_time, capacity, demand = DataLoader.generate_sample_data(N)
    
    network = DeliveryNetwork()
    network.build_from_matrix(N, flight_time, capacity, demand)
    
    drone_capacity = 20
    
    if args.problem == 1 or args.problem is None:
        num_drones = solve_problem_one(network, drone_capacity)
    
    if args.problem == 2 or args.problem is None:
        solve_problem_two(network, num_drones if 'num_drones' in locals() else 10)
    
    if args.problem == 3 or args.problem is None:
        resilience = solve_problem_three(network)
    
    if args.visualize:
        import os
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results', 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        visualizer = NetworkVisualizer(network)
        fig = visualizer.draw_network()
        fig.savefig(os.path.join(figures_dir, 'network.png'), dpi=300, bbox_inches='tight')
        
        if 'resilience' in locals():
            res_visualizer = ResilienceVisualizer(resilience)
            fig = res_visualizer.plot_resilience_curve()
            fig.savefig(os.path.join(figures_dir, 'resilience.png'), dpi=300, bbox_inches='tight')
        
        print("\n可视化图表已保存到 data/results/figures/")

if __name__ == '__main__':
    main()