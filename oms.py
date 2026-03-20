"""
订单管理系统（OMS）模块
=====================================
提供抽象接口 OrderManagementSystem 和两个实现：
  - PaperOMS   : 纸面交易（记录到 JSON 文件，不实际下单）
  - LiveOMS    : 实盘接口占位（TODO，待对接券商 API）

用法：
    from oms import PaperOMS, OrderResult

    oms = PaperOMS()
    result = oms.submit_order('0700.HK', '买入', 100, 380.0)
    print(result)
"""

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ORDERS_FILE = Path(__file__).parent / 'data' / 'logs' / 'orders.jsonl'
_ORDERS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── 数据类 ───────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """单笔订单执行结果"""
    order_id:   str
    ticker:     str
    action:     str           # '买入' | '卖出' | '观望'
    shares:     int
    price:      float
    amount:     float         # shares * price（未含费用）
    status:     str           # 'submitted' | 'rejected' | 'filled' | 'pending'
    message:    str = ''
    timestamp:  str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return (f"[{self.status}] {self.action} {self.shares}股 "
                f"{self.ticker} @ {self.price:.2f}  amount={self.amount:.2f}  "
                f"msg={self.message}")


# ── 抽象基类 ─────────────────────────────────────────────────────

class OrderManagementSystem(ABC):
    """OMS 抽象基类，所有实现必须继承此类"""

    @abstractmethod
    def submit_order(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        note: str = '',
    ) -> OrderResult:
        """
        提交订单。

        Parameters
        ----------
        ticker : 股票代码，如 '0700.HK'
        action : '买入' 或 '卖出'
        shares : 数量（整数股）
        price  : 委托价格（0 表示市价）
        note   : 附加备注

        Returns
        -------
        OrderResult
        """

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单（可选实现）"""
        logger.warning(f"cancel_order 未实现，order_id={order_id}")
        return False

    def get_position(self, ticker: str) -> dict:
        """查询持仓（可选实现）"""
        return {}


# ── PaperOMS：纸面交易 ───────────────────────────────────────────

class PaperOMS(OrderManagementSystem):
    """
    纸面交易实现：将订单记录到 JSON Lines 文件，不实际下单。
    适用于回测验证和信号推送。
    """

    def __init__(self, orders_file: Optional[Path] = None):
        self._file = orders_file or _ORDERS_FILE
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._counter = self._load_counter()

    def _load_counter(self) -> int:
        """从历史订单文件推断已有订单数（用于生成顺序 order_id）"""
        if not self._file.exists():
            return 0
        try:
            lines = self._file.read_text(encoding='utf-8').strip().splitlines()
            return len(lines)
        except Exception:
            return 0

    def submit_order(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        note: str = '',
    ) -> OrderResult:
        if action not in ('买入', '卖出'):
            return OrderResult(
                order_id='',
                ticker=ticker,
                action=action,
                shares=shares,
                price=price,
                amount=0.0,
                status='rejected',
                message=f"不支持的操作: {action}，仅支持 '买入' / '卖出'",
            )

        if shares <= 0 or price < 0:
            return OrderResult(
                order_id='',
                ticker=ticker,
                action=action,
                shares=shares,
                price=price,
                amount=0.0,
                status='rejected',
                message=f"无效参数: shares={shares}, price={price}",
            )

        self._counter += 1
        order_id = f"PAPER-{datetime.now().strftime('%Y%m%d')}-{self._counter:04d}"
        amount = shares * price

        result = OrderResult(
            order_id=order_id,
            ticker=ticker,
            action=action,
            shares=shares,
            price=price,
            amount=amount,
            status='submitted',
            message=note or f"纸面{action}，已记录",
        )

        # 追加到 JSONL 文件
        try:
            with open(self._file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            logger.warning(f"PaperOMS: 写入订单日志失败: {e}")

        logger.info(f"PaperOMS: {result}")
        return result

    def get_orders(self) -> list:
        """读取所有历史纸面订单"""
        if not self._file.exists():
            return []
        try:
            orders = []
            for line in self._file.read_text(encoding='utf-8').strip().splitlines():
                if line.strip():
                    orders.append(json.loads(line))
            return orders
        except Exception as e:
            logger.warning(f"PaperOMS: 读取订单历史失败: {e}")
            return []


# ── LiveOMS：实盘接口占位 ─────────────────────────────────────────

class LiveOMS(OrderManagementSystem):
    """
    实盘 OMS 占位实现（TODO）。
    目前只记录日志，不发送任何实际委托。

    待接入券商 API 时，在此实现 submit_order() 的真实逻辑。
    支持的 API 候选：
      - 富途 OpenAPI (moomoo)：https://openapi.futunn.com/
      - Tiger Brokers API：https://quant.tigerbrokers.com/
      - Interactive Brokers TWS API：https://interactivebrokers.github.io/tws-api/
    配置项（在 keys.yaml 中设置）：
      broker_api_url: <券商API地址>
      broker_api_key: <API密钥>
      broker_account:  <账户号>
    """

    def __init__(self, config: Optional[dict] = None):
        self._cfg = config or {}
        api_url = self._cfg.get('broker_api_url', '')
        if not api_url:
            logger.warning(
                "LiveOMS: broker_api_url 未配置，所有订单将被记录但不实际发送。"
                "请在 keys.yaml 中配置 broker_api_url / broker_api_key / broker_account。"
            )
        self._paper = PaperOMS()  # 降级为纸面记录

    def submit_order(
        self,
        ticker: str,
        action: str,
        shares: int,
        price: float,
        note: str = '',
    ) -> OrderResult:
        # TODO: 在此实现真实的券商 API 调用
        # 示例（富途 API 伪代码）：
        #   from moomoo import OpenQuoteContext, TrdSide, TrdEnv
        #   trd_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        #   trd_ctx.place_order(price=price, qty=shares, code=ticker,
        #                       trd_side=TrdSide.BUY if action=='买入' else TrdSide.SELL)
        logger.warning(
            f"LiveOMS.submit_order() 尚未实现实盘对接，"
            f"订单已降级记录为纸面交易: {action} {shares}股 {ticker} @ {price}"
        )
        result = self._paper.submit_order(ticker, action, shares, price,
                                          note=f"[LiveOMS降级] {note}")
        result.status = 'pending'
        result.message = 'LiveOMS 未配置实盘 API，已记录为纸面订单'
        return result

    def cancel_order(self, order_id: str) -> bool:
        # TODO: 实现实盘撤单
        logger.warning(f"LiveOMS.cancel_order() 未实现，order_id={order_id}")
        return False

    def get_position(self, ticker: str) -> dict:
        # TODO: 查询实盘持仓
        logger.warning(f"LiveOMS.get_position() 未实现，ticker={ticker}")
        return {}


# ── 工厂函数 ─────────────────────────────────────────────────────

def create_oms(config: Optional[dict] = None) -> OrderManagementSystem:
    """
    根据配置创建 OMS 实例。
    config 中有 broker_api_url 且非空时返回 LiveOMS，否则返回 PaperOMS。
    """
    cfg = config or {}
    if cfg.get('broker_api_url'):
        return LiveOMS(cfg)
    return PaperOMS()


# ── 单元测试 ─────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("OMS 单元测试")
    print("=" * 50)

    oms = PaperOMS(orders_file=Path('/tmp/test_orders.jsonl'))

    r1 = oms.submit_order('0700.HK', '买入', 100, 380.0, note='测试买入')
    print(f"  买入: {r1}")

    r2 = oms.submit_order('0700.HK', '卖出', 100, 395.0, note='测试卖出')
    print(f"  卖出: {r2}")

    r3 = oms.submit_order('0700.HK', '观望', 0, 0, note='观望')
    print(f"  观望（应被拒绝）: {r3}")

    orders = oms.get_orders()
    print(f"\n  历史订单（共 {len(orders)} 条）:")
    for o in orders:
        print(f"    {o['order_id']}  {o['action']}  {o['shares']}股  status={o['status']}")

    print("\n✅ PaperOMS 测试完成")

