import yaml
from typing import Any, Tuple, Dict, List

class Config:
    def __init__(self, config_dict: dict):
        """递归将字典转换为对象属性"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def build_from_yaml_file(cls, filepath: str) -> 'Config':
        """从 YAML 文件构建配置对象"""
        with open(filepath, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file) or {}
            return cls(data)

    def to_dict(self) -> Dict:
        """将Config对象递归转换回字典"""
        result = {}
        for key in self.keys():
            value = getattr(self, key)
            if isinstance(value, Config):
                # 递归转换嵌套的Config对象
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def to_yaml_file(self, filepath: str):
        """将配置转存为YAML文件"""
        with open(filepath, 'w', encoding='utf-8') as file:
            yaml.safe_dump(
                self.to_dict(),
                file,
                default_flow_style=False,  # 更易读的分块格式
                allow_unicode=True,        # 支持Unicode字符
                sort_keys=False           # 保持键的原始顺序
            )

    def keys(self) -> List[str]:
        """返回当前层级的有效键列表"""
        return [key for key in vars(self) if not key.startswith('_')]

    def values(self) -> List[Any]:
        """返回当前层级的有效值列表"""
        return [getattr(self, key) for key in self.keys()]

    def items(self) -> List[Tuple[str, Any]]:
        """返回当前层级的键值对"""
        return [(key, getattr(self, key)) for key in self.keys()]

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问 `config[key]`"""
        return getattr(self, key)

    def __repr__(self) -> str:
        """友好的对象表示"""
        return f"<Config: {self.keys()}>"

    def pop(self, key: str, default_var: Any = None):
        """pop out the key-value item from the config.

        Args:
            key (str): key name.
            default_var (Any): default value to pop.

        Returns:
            Any: value to pop.
        """
        return vars(self).pop(key, default_var)

    def get(self, key: str, default_var: Any = None):
        """Retrieve the key-value item from the config.

        Args:
            key (str): key name.
            default_var (Any): default value to pop.

        Returns:
            Any: value to get.
        """
        return vars(self).get(key, default_var)

    def set(self, key: str, var_to_set: Any):
        """Set the key-value item from the config.

        Args:
            key (str): key name.
            var_to_set (Any): default value to pop.

        Returns:
            Any: value to get.
        """
        vars(self)[key] = var_to_set

