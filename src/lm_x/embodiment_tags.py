from enum import Enum


class EmbodimentTag(Enum):
    """Robot embodiment identifiers stored in checkpoint processor metadata."""

    # Project-specific identifiers retained to load existing processor metadata.
    QinLongROS1 = "QinLongROS1"
    QinLongROS2 = "QinLongROS2"
    Genie1 = "Genie1"
    GR2 = "GR2"
    lejukuafu = "lejukuafu"
    xinghaitu_r1 = "xinghaitu_r1"
    ZhiYuanA2 = "ZhiYuanA2"
    arx_loong = "arx_loong"
    AstribotS1 = "AstribotS1"
    cobotmagic = "cobotmagic"
    fr3 = "fr3"
    TIANJI = "TIANJI"
    DualUR5e = "DualUR5e"
    Dwheel = "Dwheel"
    zhengzhou_zhiyuan_G1 = "zhengzhou_zhiyuan_G1"
    # robomind
    agilex_1cam = "agilex_1cam"
    agilex_3cam = "agilex_3cam"
    agilex_mobile = "agilex_mobile"
    franka_1cam = "franka_1cam"
    franka_3cam = "franka_3cam"
    franka_4cam = "franka_4cam"
    franka_6cam = "franka_6cam"
    tianyi = "tianyi"
    tianyi_mobile = "tianyi_mobile"
    tienkung_dim16 = "tienkung_dim16"
    tienkung_dim26 = "tienkung_dim26"
    ur5 = "ur5"
    ur5_dex = "ur5_dex"
    # V3
    sim_libero_franka = "sim_libero_franka"
    sim_robotwin2_aloha_agilex = "sim_robotwin2_aloha_agilex"
    sim_robotwin2_arx_x5 = "sim_robotwin2_arx_x5"
    sim_robotwin2_franka = "sim_robotwin2_franka"
    sim_robotwin2_piper = "sim_robotwin2_piper"
    sim_robotwin2_ur5 = "sim_robotwin2_ur5"

    ROBOCASA_GR1_TABLETOP = "robocasa_gr1_tabletop"
    """
    RoboCasa GR1 tabletop tasks with arms, waist, and Fourier hands.
    Uses the custom-embodiment projector slot.
    """

    ROBOCASA_PANDA_OMRON = "robocasa_panda_omron"
    """
    RoboCasa Panda arm tasks with an Omron gripper.
    Uses the custom-embodiment projector slot.
    """

    @classmethod
    def resolve(cls, tag: "str | EmbodimentTag") -> "EmbodimentTag":
        """Resolve a string to an EmbodimentTag, case-insensitively.

        Matches by enum **name** first (e.g. ``"xdof"`` -> ``XDOF``), then by
        enum **value** (e.g. ``"xdof_relative_eef_relative_joint"`` -> ``XDOF``).

        Raises:
            ValueError: If *tag* does not match any known embodiment.
        """
        if isinstance(tag, cls):
            return tag
        key = tag.strip()
        key_lower = key.lower()
        # Match by enum name (case-insensitive)
        for member in cls:
            if member.name.lower() == key_lower:
                return member
        # Match by enum value (case-insensitive)
        for member in cls:
            if member.value.lower() == key_lower:
                return member

        known = "\n".join(f"    {member.name:40s} -> {member.value}" for member in cls)
        raise ValueError(f"Unknown embodiment tag: {tag!r}\n\nKnown tags:\n{known}")

    @classmethod
    def reverse_lookup(cls, value: str) -> "str":
        """Map a tag value string back to its enum name, or return the value as-is."""
        for member in cls:
            if member.value == value:
                return member.name
        return value
