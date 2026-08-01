"""F3: 测试数据构造工具.

输入：数据模板 + 约束条件
输出：测试数据集（正常/边界/异常/脏数据）

流程：
    1. 指定数据模板（JSON Schema 或字段定义）
    2. LLM理解约束（字段类型、取值范围、业务规则）
    3. 生成各类测试数据
    4. 输出数据文件 + 可重复执行的生成脚本
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter


class DataType(str, Enum):
    """数据类型."""

    NORMAL = "normal"          # 正常数据
    BOUNDARY = "boundary"      # 边界值数据
    EXCEPTION = "exception"    # 异常/脏数据
    STRESS = "stress"          # 压力/大容量数据


@dataclass
class FieldDef:
    """字段定义."""

    name: str
    field_type: str            # string, integer, float, boolean, date, enum, email, phone
    required: bool = True
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    min_len: Optional[int] = None
    max_len: Optional[int] = None
    enum_values: List[str] = field(default_factory=list)
    pattern: str = ""          # 正则约束
    description: str = ""


@dataclass
class DataTemplate:
    """数据模板."""

    name: str
    fields: List[FieldDef] = field(default_factory=list)
    description: str = ""

    def to_prompt_str(self) -> str:
        """转为 Prompt 中的字段描述."""
        lines = [f"数据模板: {self.name}"]
        if self.description:
            lines.append(f"描述: {self.description}")
        lines.append("字段定义:")
        for f in self.fields:
            parts = [f"  - {f.name} ({f.field_type})"]
            if f.required:
                parts.append("必填")
            else:
                parts.append("选填")
            if f.min_val is not None:
                parts.append(f"最小值={f.min_val}")
            if f.max_val is not None:
                parts.append(f"最大值={f.max_val}")
            if f.min_len is not None:
                parts.append(f"最小长度={f.min_len}")
            if f.max_len is not None:
                parts.append(f"最大长度={f.max_len}")
            if f.enum_values:
                parts.append(f"可选值={f.enum_values}")
            if f.pattern:
                parts.append(f"格式={f.pattern}")
            if f.description:
                parts.append(f"说明: {f.description}")
            lines.append(" ".join(parts))
        return "\n".join(lines)


@dataclass
class GeneratedData:
    """单条生成数据."""

    data_id: str
    data_type: DataType
    record: Dict[str, Any]
    reason: str = ""           # 生成该数据的理由

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_id": self.data_id,
            "data_type": self.data_type.value,
            "record": self.record,
            "reason": self.reason,
        }


@dataclass
class DataFactoryReport:
    """数据构造报告."""

    template_name: str = ""
    total_records: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    records: List[GeneratedData] = field(default_factory=list)
    generation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "template_name": self.template_name,
            "total_records": self.total_records,
            "by_type": self.by_type,
            "records": [r.to_dict() for r in self.records],
            "generation_time_ms": self.generation_time_ms,
        }


# 数据生成 Prompt
_DATA_GEN_PROMPT = """你是一位测试数据工程师，请根据以下数据模板生成测试数据。

{template_desc}

请生成以下四类测试数据：
1. normal（正常数据）：符合所有约束的有效数据，生成{normal_count}条
2. boundary（边界值数据）：取值在边界上的数据，如最大值、最小值、空字符串、极大值，生成{boundary_count}条
3. exception（异常/脏数据）：类型错误、格式错误、缺失必填字段、注入攻击字符串，生成{exception_count}条
4. stress（压力数据）：超大字符串、超长列表、深层嵌套，生成{stress_count}条

每条数据格式如下（JSON数组）：
```json
[
  {{
    "data_type": "normal|boundary|exception|stress",
    "record": {{"field1": "value1", "field2": "value2"}},
    "reason": "生成理由"
  }}
]
```

要求：
- 正常数据必须严格符合约束
- 边界值要覆盖所有字段的边界
- 异常数据要包含各类典型脏数据（SQL注入、XSS、超长字符串、类型错误等）
- 只输出JSON，不要其他内容
"""

# 内置数据模板
BUILTIN_TEMPLATES = {
    "user_profile": DataTemplate(
        name="user_profile",
        description="用户注册信息",
        fields=[
            FieldDef(name="username", field_type="string", required=True, min_len=3, max_len=20, pattern="^[a-zA-Z0-9_]+$"),
            FieldDef(name="email", field_type="email", required=True),
            FieldDef(name="age", field_type="integer", required=True, min_val=1, max_val=150),
            FieldDef(name="phone", field_type="phone", required=False, pattern="^1[3-9]\\d{9}$"),
            FieldDef(name="gender", field_type="enum", required=False, enum_values=["male", "female", "other"]),
        ],
    ),
    "product": DataTemplate(
        name="product",
        description="商品信息",
        fields=[
            FieldDef(name="product_id", field_type="string", required=True, pattern="^SKU-\\d{6}$"),
            FieldDef(name="name", field_type="string", required=True, min_len=1, max_len=100),
            FieldDef(name="price", field_type="float", required=True, min_val=0.01, max_val=999999.99),
            FieldDef(name="stock", field_type="integer", required=True, min_val=0, max_val=100000),
            FieldDef(name="category", field_type="enum", required=True, enum_values=["electronics", "clothing", "food", "books", "other"]),
            FieldDef(name="description", field_type="string", required=False, max_len=500),
        ],
    ),
    "order": DataTemplate(
        name="order",
        description="订单信息",
        fields=[
            FieldDef(name="order_id", field_type="string", required=True, pattern="^ORD-\\d{12}$"),
            FieldDef(name="user_id", field_type="string", required=True),
            FieldDef(name="amount", field_type="float", required=True, min_val=0.01),
            FieldDef(name="status", field_type="enum", required=True, enum_values=["pending", "paid", "shipped", "delivered", "cancelled"]),
            FieldDef(name="items", field_type="integer", required=True, min_val=1, max_val=100),
            FieldDef(name="address", field_type="string", required=True, min_len=5),
        ],
    ),
}


class DataFactory:
    """测试数据构造器."""

    def __init__(self, adapter: BaseModelAdapter) -> None:
        self._adapter = adapter

    def _parse_data(self, raw: str) -> List[GeneratedData]:
        """解析LLM生成的数据."""
        # 优先提取 ```json ... ``` 代码块
        code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', raw)
        json_str = code_block.group(1) if code_block else raw

        arr_start = json_str.find('[')
        arr_end = json_str.rfind(']')
        if arr_start == -1 or arr_end == -1 or arr_end <= arr_start:
            return []

        try:
            items = json.loads(json_str[arr_start:arr_end + 1])
        except json.JSONDecodeError:
            return []

        records = []
        for i, item in enumerate(items):
            dt_str = item.get("data_type", "normal")
            try:
                data_type = DataType(dt_str)
            except ValueError:
                data_type = DataType.NORMAL

            record = GeneratedData(
                data_id=f"data_{i+1:03d}",
                data_type=data_type,
                record=item.get("record", {}),
                reason=item.get("reason", ""),
            )
            records.append(record)

        return records

    def generate(
        self,
        template: DataTemplate,
        normal_count: int = 3,
        boundary_count: int = 3,
        exception_count: int = 3,
        stress_count: int = 2,
    ) -> DataFactoryReport:
        """根据模板生成测试数据."""
        import time

        start = time.time()
        report = DataFactoryReport(template_name=template.name)

        prompt = _DATA_GEN_PROMPT.format(
            template_desc=template.to_prompt_str(),
            normal_count=normal_count,
            boundary_count=boundary_count,
            exception_count=exception_count,
            stress_count=stress_count,
        )

        result = self._adapter.generate(prompt, max_tokens=2048)
        records = self._parse_data(result.text)

        report.records = records
        report.total_records = len(records)
        for r in records:
            dt = r.data_type.value
            report.by_type[dt] = report.by_type.get(dt, 0) + 1

        report.generation_time_ms = (time.time() - start) * 1000
        return report

    def generate_from_json_schema(
        self, schema: Dict[str, Any], name: str = "custom"
    ) -> DataFactoryReport:
        """从 JSON Schema 生成数据."""
        fields = []
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        for fname, fdef in properties.items():
            ftype = fdef.get("type", "string")
            # JSON Schema type 映射
            type_map = {"string": "string", "integer": "integer", "number": "float", "boolean": "boolean"}
            mapped_type = type_map.get(ftype, "string")

            field = FieldDef(
                name=fname,
                field_type=mapped_type,
                required=fname in required_fields,
                min_val=fdef.get("minimum"),
                max_val=fdef.get("maximum"),
                min_len=fdef.get("minLength"),
                max_len=fdef.get("maxLength"),
                enum_values=fdef.get("enum", []),
                pattern=fdef.get("pattern", ""),
                description=fdef.get("description", ""),
            )
            fields.append(field)

        template = DataTemplate(name=name, fields=fields, description=schema.get("description", ""))
        return self.generate(template)

    def save_report(self, report: DataFactoryReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"data_{report.template_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def print_report(report: DataFactoryReport) -> None:
        """打印报告."""
        print(f"\n测试数据构造报告")
        print(f"  模板: {report.template_name}")
        print(f"  生成数据条数: {report.total_records}")
        print(f"  耗时: {report.generation_time_ms:.0f}ms")

        print(f"\n  按类型分布:")
        for dt, count in sorted(report.by_type.items()):
            print(f"    {dt}: {count}")

        # 展示部分数据
        for r in report.records[:8]:
            print(f"\n  [{r.data_type.value}] {r.data_id}: {json.dumps(r.record, ensure_ascii=False)[:80]}")
            if r.reason:
                print(f"    理由: {r.reason}")

        if report.total_records > 8:
            print(f"\n  ... 共{report.total_records}条，仅展示前8条")
