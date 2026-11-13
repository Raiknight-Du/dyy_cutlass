# S组有限screen条件准入

Lead已独立复算原始SASS，并明确批准本组screen；此文件不替代STATIC_GATE_REVIEW中的限制，不标记绝对局部无偏。

唯一FRAME哈希：710d99de197cf0c4e69ec9e4c5a6a579ad0132b351e53efc8b1d243a75065223。B0=06ebd78963371a30adac142c06b86950ec300b6cb762a209c5afc67628941814，Pbase=2d67694769e415d2aba5323fcdc6efab420b3dc1a851dd99ef4669b9461aaa4a。旧FRAME 8f9d0584…因W8谓词错误已撤销，未启动。

INIT的328条指令只有3处uniform live-in名变化：UR20→UR14一处、UR31→UR35两处。后二者都来自S2UR SR_CTAID.Y。前者为原get_cp_work地址计算传入state路径的live-in，源码多值赋值、参数和继承helper完全未改；编译前导整数/谓词调度改变，所以不声称整个preheader逐寄存器RAW完全等价。普通R寄存器、LDG/STTM、barrier、clock及指令顺序保持；INIT_END→首次CG1返回631条连uniform操作数亦相同。Pbase/frame0同批相对INIT稳定性须另核±5%，不能由总Event±2%替代。

AST late路径位于原CG1 if的orelse链，表示 not(warp4..7) AND warp==8；不是warp4..7 AND warp8。新导出已保留orelse否定。早PF对应SASS0x09f0把warp!=8与mask bit4合并，0x0af0跳过，因此only W8且bit4清零落入；late只有W8，TMEM rendezvous后等full phase0再PF。每CTA仅一套5条PF，后续NCU再核动态早/晚PC计数。

执行限制：全局守卫连续3×30s确认自然空闲；授权1845MHz租约结束恢复。先4buffer逐bit O/state、每变体首次调用30s watchdog；之后3warm/3ABBA有限样本。配置仅S32768 B0/Pbase/frame0/frame7，无V、无正式3进程、无NCU。输出Event、Pbase/frame0 INIT配对、frame0/7 INIT与同调用首CG1推进。screen结果再报Lead裁决，不自动进入formal。

model_enabled=false；局部绝对bias=null。
