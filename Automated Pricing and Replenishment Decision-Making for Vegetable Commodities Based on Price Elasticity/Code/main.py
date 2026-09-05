import sys
from pathlib import Path
import time
import pandas as pd

SRC=Path(__file__).resolve().parent/"src"
sys.path.insert(0,str(SRC))

from config import OUTPUT_DIR
from data_io import load_all, prepare_sales
from problem1 import build_tables as p1_tables, save_figures as p1_figures
from problem2 import forecast_categories, price_elasticity, optimize_pricing, save_figures as p2_figures
from problem3 import build_candidates, solve, summary, sensitivity, save_figures as p3_figures
from report import save_tables, save_run_summary


def main():
	t0=time.perf_counter()
	print("开始运行 C题建模代码")
	
	goods,sales,wholesale,loss_cat,loss_item=load_all()
	print(f"数据读取完成：销售流水 {len(sales):,} 行")
	
	sales=prepare_sales(sales,	)
	print(f"数据清洗完成：有效销售 {len(sales):,} 行")

	# 问题一
	t=time.perf_counter()
	p1=p1_tables(sales,wholesale)
	
	p1_figures(sales,p1)
	print(f"问题一完成，用时 {time.perf_counter()-t:.1f}s")

	# 问题二
	t=time.perf_counter()
	fc,valid=forecast_categories(p1["daily"])
	elast=price_elasticity(sales)
	pricing,detail=optimize_pricing(fc,elast,wholesale,loss_cat,sales,goods)
	p2_figures(fc,elast,pricing)
	print(f"问题二完成，用时 {time.perf_counter()-t:.1f}s")

	# 问题三
	t=time.perf_counter()
	candidates,factor=build_candidates(sales,wholesale,goods,loss_item,pricing)
	day1_forecast = fc.iloc[0]
	chosen,res=solve(candidates,day1_forecast)
	cov=summary(chosen,day1_forecast)
	sens=sensitivity(candidates,day1_forecast)
	p3_figures(cov)
	print(f"问题三完成：候选 {len(candidates)}，选中 {len(chosen)}，用时 {time.perf_counter()-t:.1f}s")

	# 问题四：论文中的数据采集建议，直接结构化输出
	p4=pd.DataFrame([
		["天气与温度数据","提升需求预测精度","温度骤降、降雨等会改变消费需求，可降低预测误差"],
		["节假日与促销记录","刻画需求突发波动","春节、节假日和促销活动会形成需求脉冲"],
		["客流与顾客画像数据","改善销量关联分析","客流可解释品类间同步波动，画像可细分需求"],
		["详细损耗与到期记录","精确损耗模型","将品类平均损耗细化到单品和时间维度"],
		["缺货与脱销记录","修正需求预测","缺货会使观察到的销量低于真实需求"],
		["竞争商超价格","定价约束更真实","用于设置更合理的加成率上界"],
		["供应商与物流数据","优化补货可行性","到货时间、稳定性和质量会影响补货约束"],
		["采购批量与折扣","细化成本模型","批量折扣会改变单位采购成本"],
	],columns=["数据","对问题的帮助","理由"])

	tables={
		"表2_品类日销量分布统计":p1["table2"],"表3_星期效应":p1["table3"],"表4_月度效应":p1["table4"],"表5_Pearson相关矩阵":p1["table5"],
		"表6_需求预测留出验证":valid,"表7_未来7日需求预测":fc.reset_index().rename(columns={"index":"日期"}),"表8_价格弹性":elast,
		"表9_问题二优化结果":detail,"表9_品类最优加成率":pricing,"表10_问题三品类覆盖":cov,
		"表11_问题三单品结果":chosen[["单品编码","单品名称","分类名称","预测需求(kg)","补货量(kg)","批发价","损耗率(%)","售价(元/kg)","加成率","利润(元)"]],
		"表12_敏感性分析":sens,"表13_补充数据采集建议":p4
	}
	save_tables(tables)
	summary_text=(f"运行完成。总耗时 {time.perf_counter()-t0:.1f} 秒。\n"
				  f"输出目录：{OUTPUT_DIR}\n"
				  f"问题三选中单品数：{len(chosen)}\n"
				  f"问题三总利润：{chosen['利润(元)'].sum():.2f} 元\n"
				  "首次运行会将大 Excel 转换为 Parquet 缓存；以后运行可显著减少读取时间。")
	save_run_summary(summary_text)
	print(summary_text)
	print("运行结束")

if __name__=="__main__": main()
