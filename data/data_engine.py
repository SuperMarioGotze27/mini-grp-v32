#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mini-GRP 数据采集模块
使用 akshare 获取 A 股数据，包含财务指标、价格数据和行业分类

作者: Quant Dev
日期: 2025-06-09
"""

import logging
import time
import random
from typing import List, Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 重试装饰器
# ---------------------------------------------------------------------------

def retry_on_failure(max_retries=3, delay=2, backoff=2):
    """为函数添加重试机制的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries >= max_retries:
                        logger.error(
                            f"函数 {func.__name__} 在 {max_retries} 次尝试后仍然失败: {e}"
                        )
                        raise
                    logger.warning(
                        f"函数 {func.__name__} 第 {retries} 次尝试失败: {e}, "
                        f"{current_delay}秒后重试..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
            return None
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 公共工具函数
# ---------------------------------------------------------------------------

def _safe_get(func, *args, **kwargs):
    """安全地调用 akshare 函数，失败时返回 None"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"akshare 调用失败 {func.__name__}: {e}")
        return None


def _generate_mock_stock_list(n: int = 100) -> pd.DataFrame:
    """生成模拟股票列表（用于网络不可用的测试环境）"""
    logger.info(f"生成 {n} 只模拟股票数据用于测试...")
    np.random.seed(42)
    industries = [
        '银行', '非银金融', '医药生物', '电子', '食品饮料',
        '电力设备', '计算机', '汽车', '化工', '机械设备',
        '家用电器', '通信', '房地产', '有色金属', '传媒',
        '交通运输', '农林牧渔', '建筑装饰', '钢铁', '采掘'
    ]
    stocks = []
    for i in range(n):
        code = f"{600000 + i * 3:06d}" if i < 50 else f"{300001 + (i - 50) * 3:06d}"
        if i < 50:
            code = f"{600000 + i * 7:06d}"
        else:
            code = f"{300001 + (i - 50) * 7:06d}"
        # 确保代码有效性
        code = f"{code[:2]}{int(code[2:]):04d}"
        stocks.append({
            'code': code,
            'name': f"股票_{i+1}",
            'industry': random.choice(industries),
            'list_date': (datetime.now() - timedelta(days=random.randint(365, 3650))).strftime('%Y-%m-%d')
        })
    return pd.DataFrame(stocks)


def _generate_mock_financials(stock_codes: List[str]) -> pd.DataFrame:
    """生成模拟财务指标数据"""
    logger.info(f"生成 {len(stock_codes)} 只股票的模拟财务数据...")
    np.random.seed(43)
    data = []
    for code in stock_codes:
        pe = np.random.normal(25, 15)
        pb = np.random.normal(2.5, 1.5)
        data.append({
            'code': code,
            'name': f"股票_{code}",
            'pe_ttm': max(pe, 1.0),
            'pb_lf': max(pb, 0.5),
            'ps_ttm': max(np.random.normal(3, 2), 0.3),
            'ev_ebitda': max(np.random.normal(15, 10), 2.0),
            'roe_deducted': np.random.normal(10, 8),
            'roa': np.random.normal(5, 4),
            'gross_margin': np.random.normal(30, 15),
            'net_margin': np.random.normal(12, 10),
            'debt_to_equity': max(np.random.normal(60, 30), 5.0),
            'current_ratio': max(np.random.normal(1.5, 0.8), 0.3),
            'fcf_yield': np.random.normal(3, 2),
            'revenue_yoy': np.random.normal(15, 20),
            'profit_yoy': np.random.normal(12, 25),
            'eps_growth_3y': np.random.normal(10, 15),
            'total_mv': np.random.uniform(50, 5000),
            'turnover_20d': np.random.uniform(1, 10),
            'dividend_yield': max(np.random.normal(2, 1.5), 0.0),
        })
    return pd.DataFrame(data)


def _generate_mock_prices(stock_codes: List[str]) -> pd.DataFrame:
    """生成模拟价格数据"""
    logger.info(f"生成 {len(stock_codes)} 只股票的模拟价格数据...")
    np.random.seed(44)
    data = []
    for code in stock_codes:
        data.append({
            'code': code,
            'return_1m': np.random.normal(0, 8),
            'return_3m': np.random.normal(0, 15),
            'return_6m': np.random.normal(0, 25),
            'return_12m': np.random.normal(10, 35),
            'volatility_20d': np.random.uniform(15, 50),
            'avg_volume_20d': np.random.uniform(1000, 100000),
        })
    return pd.DataFrame(data)


def _generate_mock_industries(stock_codes: List[str]) -> pd.DataFrame:
    """生成模拟行业分类数据"""
    logger.info(f"生成 {len(stock_codes)} 只股票的模拟行业分类...")
    np.random.seed(45)
    industries = [
        '银行', '非银金融', '医药生物', '电子', '食品饮料',
        '电力设备', '计算机', '汽车', '化工', '机械设备',
        '家用电器', '通信', '房地产', '有色金属', '传媒',
        '交通运输', '农林牧渔', '建筑装饰', '钢铁', '采掘'
    ]
    data = []
    for code in stock_codes:
        data.append({
            'code': code,
            'sw_industry_name': random.choice(industries)
        })
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 核心数据获取函数
# ---------------------------------------------------------------------------

def fetch_stock_list() -> pd.DataFrame:
    """
    获取 A 股所有上市公司列表

    Returns:
        DataFrame with columns [code, name, industry, list_date]
        使用 ak.stock_zh_a_spot_em() 获取全市场股票
    """
    logger.info("开始获取 A 股上市公司列表...")

    try:
        import akshare as ak
        df = _safe_get(ak.stock_zh_a_spot_em)

        if df is None or df.empty:
            logger.warning("akshare 返回空数据，使用模拟数据")
            return _generate_mock_stock_list(100)

        # 统一列名映射（东方财富字段 -> 标准字段）
        column_mapping = {
            '代码': 'code',
            '名称': 'name',
            '行业': 'industry',
        }

        result = pd.DataFrame()
        for cn_col, en_col in column_mapping.items():
            if cn_col in df.columns:
                result[en_col] = df[cn_col]

        # 如果没有找到标准列名，尝试使用通用列名
        if result.empty and len(df.columns) >= 2:
            result['code'] = df.iloc[:, 1]  # 通常第2列是代码
            result['name'] = df.iloc[:, 2]  # 通常第3列是名称

        # 添加上市日期（模拟）
        result['list_date'] = ''

        # 过滤无效代码
        result = result[result['code'].notna()]
        result = result[result['code'].astype(str).str.match(r'^\d{6}$', na=False)]

        logger.info(f"成功获取 {len(result)} 只股票")
        return result.reset_index(drop=True)

    except ImportError:
        logger.warning("akshare 未安装，使用模拟数据")
        return _generate_mock_stock_list(100)
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return _generate_mock_stock_list(100)


def fetch_financial_indicators(stock_codes: List[str]) -> pd.DataFrame:
    """
    获取指定股票的财务指标

    使用 ak.stock_zh_a_spot_em() 获取市场数据，包含 PE、PB、ROE 等
    或者使用 ak.stock_financial_analysis_indicator() 获取详细财务指标

    Args:
        stock_codes: 股票代码列表

    Returns:
        DataFrame with columns:
            [code, name, pe_ttm, pb_lf, ps_ttm, ev_ebitda,
             roe_deducted, roa, gross_margin, net_margin,
             debt_to_equity, current_ratio, fcf_yield,
             revenue_yoy, profit_yoy, eps_growth_3y,
             total_mv, turnover_20d, dividend_yield]
    """
    if not stock_codes:
        logger.warning("股票代码列表为空，返回空 DataFrame")
        return pd.DataFrame()

    logger.info(f"开始获取 {len(stock_codes)} 只股票的财务指标...")

    try:
        import akshare as ak

        # 首先尝试获取全市场快照数据
        spot_df = _safe_get(ak.stock_zh_a_spot_em)

        if spot_df is None or spot_df.empty:
            logger.warning("无法获取市场快照数据，使用模拟数据")
            return _generate_mock_financials(stock_codes[:100])

        # 构建结果 DataFrame
        result_rows = []

        # 东方财富字段映射
        em_mapping = {
            '代码': 'code',
            '名称': 'name',
            '市盈率-动态': 'pe_ttm',
            '市净率': 'pb_lf',
            '市销率': 'ps_ttm',
            '总市值': 'total_mv',
            '换手率': 'turnover_20d',
        }

        for code in stock_codes:
            try:
                # 从 spot 数据中查找
                row_data = spot_df[spot_df['代码'] == code]
                if row_data.empty:
                    continue

                info = {'code': code}

                # 名称
                info['name'] = row_data['名称'].values[0] if '名称' in row_data.columns else ''

                # 市盈率
                if '市盈率-动态' in row_data.columns:
                    info['pe_ttm'] = pd.to_numeric(row_data['市盈率-动态'].values[0], errors='coerce')
                elif '市盈率' in row_data.columns:
                    info['pe_ttm'] = pd.to_numeric(row_data['市盈率'].values[0], errors='coerce')
                else:
                    info['pe_ttm'] = np.nan

                # 市净率
                if '市净率' in row_data.columns:
                    info['pb_lf'] = pd.to_numeric(row_data['市净率'].values[0], errors='coerce')
                else:
                    info['pb_lf'] = np.nan

                # 市销率
                if '市销率' in row_data.columns:
                    info['ps_ttm'] = pd.to_numeric(row_data['市销率'].values[0], errors='coerce')
                else:
                    info['ps_ttm'] = np.nan

                # 总市值（转换为亿元）
                if '总市值' in row_data.columns:
                    mv = pd.to_numeric(row_data['总市值'].values[0], errors='coerce')
                    info['total_mv'] = mv / 1e8 if pd.notna(mv) else np.nan
                else:
                    info['total_mv'] = np.nan

                # 换手率
                if '换手率' in row_data.columns:
                    info['turnover_20d'] = pd.to_numeric(row_data['换手率'].values[0], errors='coerce')
                else:
                    info['turnover_20d'] = np.nan

                # 对于无法从 spot 获取的指标，设为空值
                info['ev_ebitda'] = np.nan
                info['roe_deducted'] = np.nan
                info['roa'] = np.nan
                info['gross_margin'] = np.nan
                info['net_margin'] = np.nan
                info['debt_to_equity'] = np.nan
                info['current_ratio'] = np.nan
                info['fcf_yield'] = np.nan
                info['revenue_yoy'] = np.nan
                info['profit_yoy'] = np.nan
                info['eps_growth_3y'] = np.nan
                info['dividend_yield'] = np.nan

                result_rows.append(info)

            except Exception as e:
                logger.warning(f"处理股票 {code} 时出错: {e}")
                continue

        if not result_rows:
            logger.warning("未能获取任何财务数据，使用模拟数据")
            return _generate_mock_financials(stock_codes[:100])

        result = pd.DataFrame(result_rows)

        # 尝试获取更详细的财务指标（逐个股票获取，限制数量）
        if len(stock_codes) <= 50:
            for i, code in enumerate(stock_codes):
                try:
                    detail = _safe_get(
                        ak.stock_financial_analysis_indicator,
                        symbol=code,
                        start_year=str(datetime.now().year - 2)
                    )
                    if detail is not None and not detail.empty:
                        # 提取最新一期数据
                        latest = detail.iloc[0] if isinstance(detail, pd.DataFrame) else detail

                        idx = result[result['code'] == code].index
                        if len(idx) > 0:
                            # 更新财务指标
                            if '净资产收益率-摊薄(%)' in detail.columns:
                                result.loc[idx, 'roe_deducted'] = pd.to_numeric(
                                    detail['净资产收益率-摊薄(%)'].iloc[0], errors='coerce'
                                )
                            if '总资产报酬率(%)' in detail.columns:
                                result.loc[idx, 'roa'] = pd.to_numeric(
                                    detail['总资产报酬率(%)'].iloc[0], errors='coerce'
                                )
                            if '销售毛利率(%)' in detail.columns:
                                result.loc[idx, 'gross_margin'] = pd.to_numeric(
                                    detail['销售毛利率(%)'].iloc[0], errors='coerce'
                                )
                            if '销售净利率(%)' in detail.columns:
                                result.loc[idx, 'net_margin'] = pd.to_numeric(
                                    detail['销售净利率(%)'].iloc[0], errors='coerce'
                                )
                            if '资产负债率(%)' in detail.columns:
                                result.loc[idx, 'debt_to_equity'] = pd.to_numeric(
                                    detail['资产负债率(%)'].iloc[0], errors='coerce'
                                )

                    if (i + 1) % 10 == 0:
                        logger.info(f"已获取 {i+1}/{len(stock_codes)} 只股票的详细财务数据...")
                    time.sleep(0.3)  # 避免请求过快

                except Exception as e:
                    logger.warning(f"获取股票 {code} 详细财务数据失败: {e}")
                    continue

        logger.info(f"成功获取 {len(result)} 只股票的财务指标")
        return result

    except ImportError:
        logger.warning("akshare 未安装，使用模拟数据")
        return _generate_mock_financials(stock_codes[:100])
    except Exception as e:
        logger.error(f"获取财务指标失败: {e}")
        return _generate_mock_financials(stock_codes[:100])


def fetch_price_data(stock_codes: List[str],
                     end_date: Optional[str] = None) -> pd.DataFrame:
    """
    获取价格数据用于动量计算

    使用 ak.stock_zh_a_hist() 获取历史行情
    计算 1个月、3个月、6个月、12个月的收益率以及20日波动率

    Args:
        stock_codes: 股票代码列表（建议限制在500-800只）
        end_date: 结束日期，格式 'YYYYMMDD'，默认今天

    Returns:
        DataFrame with columns:
            [code, return_1m, return_3m, return_6m, return_12m,
             volatility_20d, avg_volume_20d]
    """
    if not stock_codes:
        logger.warning("股票代码列表为空，返回空 DataFrame")
        return pd.DataFrame()

    # 限制股票数量，避免数据量过大
    if len(stock_codes) > 500:
        logger.warning(f"股票数量 {len(stock_codes)} 超过500，截取前500只")
        stock_codes = stock_codes[:500]

    logger.info(f"开始获取 {len(stock_codes)} 只股票的价格数据...")

    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')

    # 计算各期起点（约250个交易日为一年）
    periods = {
        '1m': 22,
        '3m': 66,
        '6m': 132,
        '12m': 250
    }

    # 计算起始日期（预留足够的历史数据）
    start_dt = datetime.strptime(end_date, '%Y%m%d') - timedelta(days=400)
    start_date = start_dt.strftime('%Y%m%d')

    result_rows = []

    try:
        import akshare as ak

        for i, code in enumerate(stock_codes):
            try:
                # 获取历史行情
                hist = _safe_get(
                    ak.stock_zh_a_hist,
                    symbol=code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"  # 前复权
                )

                if hist is None or hist.empty or len(hist) < 30:
                    continue

                # 确保日期列是 datetime 类型
                hist['日期'] = pd.to_datetime(hist['日期'])
                hist = hist.sort_values('日期')

                # 获取收盘价
                closes = hist['收盘'].astype(float)
                volumes = hist['成交量'].astype(float) if '成交量' in hist.columns else pd.Series([0] * len(hist))

                info = {'code': code}

                # 计算各期收益率
                latest_close = closes.iloc[-1]

                for period_name, period_days in periods.items():
                    if len(closes) >= period_days:
                        past_close = closes.iloc[-period_days]
                        ret = (latest_close / past_close - 1) * 100
                    elif len(closes) >= 2:
                        # 数据不足时使用最早可用的
                        past_close = closes.iloc[0]
                        ret = (latest_close / past_close - 1) * 100
                    else:
                        ret = np.nan
                    info[f'return_{period_name}'] = ret

                # 计算20日波动率（年化）
                if len(closes) >= 22:
                    recent_returns = closes.iloc[-22:].pct_change().dropna()
                    volatility = recent_returns.std() * np.sqrt(252) * 100
                else:
                    volatility = np.nan
                info['volatility_20d'] = volatility

                # 20日平均成交量
                if len(volumes) >= 20:
                    avg_vol = volumes.iloc[-20:].mean()
                else:
                    avg_vol = volumes.mean()
                info['avg_volume_20d'] = avg_vol

                result_rows.append(info)

                if (i + 1) % 50 == 0:
                    logger.info(f"已获取 {i+1}/{len(stock_codes)} 只股票的价格数据...")

                # 添加随机延迟，避免请求过快
                time.sleep(random.uniform(0.1, 0.3))

            except Exception as e:
                logger.warning(f"获取股票 {code} 价格数据失败: {e}")
                continue

        if not result_rows:
            logger.warning("未能获取任何价格数据，使用模拟数据")
            return _generate_mock_prices(stock_codes)

        result = pd.DataFrame(result_rows)
        logger.info(f"成功获取 {len(result)} 只股票的价格数据")
        return result

    except ImportError:
        logger.warning("akshare 未安装，使用模拟数据")
        return _generate_mock_prices(stock_codes)
    except Exception as e:
        logger.error(f"获取价格数据失败: {e}")
        return _generate_mock_prices(stock_codes)


def fetch_industry_classification() -> pd.DataFrame:
    """
    获取行业分类数据

    使用 ak.stock_board_industry_name_ths() 获取同花顺行业分类
    或者从 spot_em 数据中提取

    Returns:
        DataFrame with columns [code, sw_industry_name]
    """
    logger.info("开始获取行业分类数据...")

    try:
        import akshare as ak

        # 尝试获取同花顺行业分类
        try:
            industry_df = _safe_get(ak.stock_board_industry_name_ths)
            if industry_df is not None and not industry_df.empty:
                logger.info(f"获取到 {len(industry_df)} 个行业")
                # 获取每个行业的成分股
                all_stocks = []
                for _, row in industry_df.head(30).iterrows():  # 限制行业数量
                    try:
                        industry_name = row.get('名称', '')
                        stocks = _safe_get(
                            ak.stock_board_industry_cons_ths,
                            symbol=industry_name
                        )
                        if stocks is not None and not stocks.empty and '代码' in stocks.columns:
                            for _, s in stocks.iterrows():
                                all_stocks.append({
                                    'code': s['代码'],
                                    'sw_industry_name': industry_name
                                })
                        time.sleep(0.2)
                    except Exception as e:
                        continue

                if all_stocks:
                    result = pd.DataFrame(all_stocks)
                    result = result.drop_duplicates(subset=['code'])
                    logger.info(f"成功获取 {len(result)} 只股票的行业分类")
                    return result
        except Exception as e:
            logger.warning(f"同花顺行业分类获取失败: {e}")

        # 回退：从 spot 数据中提取
        spot_df = _safe_get(ak.stock_zh_a_spot_em)
        if spot_df is not None and not spot_df.empty and '行业' in spot_df.columns:
            result = spot_df[['代码', '行业']].copy()
            result.columns = ['code', 'sw_industry_name']
            result = result.dropna(subset=['code', 'sw_industry_name'])
            logger.info(f"从行情数据中提取到 {len(result)} 只股票的行业分类")
            return result

        logger.warning("无法获取行业分类数据，使用模拟数据")
        return _generate_mock_industries([f"{i:06d}" for i in range(100)])

    except ImportError:
        logger.warning("akshare 未安装，使用模拟数据")
        return _generate_mock_industries([f"{i:06d}" for i in range(100)])
    except Exception as e:
        logger.error(f"获取行业分类失败: {e}")
        return _generate_mock_industries([f"{i:06d}" for i in range(100)])


# ---------------------------------------------------------------------------
# 数据合并工具
# ---------------------------------------------------------------------------

def merge_data(financials: pd.DataFrame,
               prices: pd.DataFrame,
               industries: pd.DataFrame) -> pd.DataFrame:
    """
    合并财务数据、价格数据和行业分类数据

    Args:
        financials: 财务指标 DataFrame
        prices: 价格数据 DataFrame
        industries: 行业分类 DataFrame

    Returns:
        合并后的 DataFrame
    """
    logger.info("开始合并数据...")

    if financials.empty:
        logger.error("财务数据为空")
        return pd.DataFrame()

    # 以财务数据为基准
    result = financials.copy()

    # 合并价格数据
    if not prices.empty:
        price_cols = ['code', 'return_1m', 'return_3m', 'return_6m', 'return_12m',
                      'volatility_20d', 'avg_volume_20d']
        price_merge = prices[[c for c in price_cols if c in prices.columns]]
        result = result.merge(price_merge, on='code', how='left')
        logger.info(f"合并价格数据: {len(result)} 行")

    # 合并行业分类
    if not industries.empty:
        industry_merge = industries[['code', 'sw_industry_name']].drop_duplicates('code')
        result = result.merge(industry_merge, on='code', how='left')
        logger.info(f"合并行业数据: {len(result)} 行")

    logger.info(f"数据合并完成，共 {len(result)} 只股票，{len(result.columns)} 列")
    return result


# ---------------------------------------------------------------------------
# 测试入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("Mini-GRP 数据采集模块测试")
    print("=" * 60)

    # 1. 测试获取股票列表
    print("\n[1/4] 测试 fetch_stock_list()...")
    stocks = fetch_stock_list()
    print(f"获取到 {len(stocks)} 只股票")
    print(stocks.head(3).to_string())

    # 2. 测试获取财务指标
    print("\n[2/4] 测试 fetch_financial_indicators()...")
    test_codes = stocks['code'].head(10).tolist()
    financials = fetch_financial_indicators(test_codes)
    print(f"获取到 {len(financials)} 只股票的财务数据")
    if not financials.empty:
        print(f"财务指标列: {financials.columns.tolist()}")
        print(financials[['code', 'name', 'pe_ttm', 'pb_lf']].head(3).to_string())

    # 3. 测试获取价格数据
    print("\n[3/4] 测试 fetch_price_data()...")
    prices = fetch_price_data(test_codes)
    print(f"获取到 {len(prices)} 只股票的价格数据")
    if not prices.empty:
        print(f"价格数据列: {prices.columns.tolist()}")
        print(prices[['code', 'return_1m', 'return_3m', 'return_12m']].head(3).to_string())

    # 4. 测试获取行业分类
    print("\n[4/4] 测试 fetch_industry_classification()...")
    industries = fetch_industry_classification()
    print(f"获取到 {len(industries)} 只股票的行业分类")
    print(industries.head(3).to_string())

    # 5. 测试数据合并
    print("\n[5/5] 测试 merge_data()...")
    merged = merge_data(financials, prices, industries)
    print(f"合并后数据: {len(merged)} 行, {len(merged.columns)} 列")
    print(f"列名: {merged.columns.tolist()}")

    print("\n" + "=" * 60)
    print("数据采集模块测试完成!")
    print("=" * 60)
