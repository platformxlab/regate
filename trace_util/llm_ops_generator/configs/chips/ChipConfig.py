from pydantic import BaseModel
from pydantic import TypeAdapter


# @dataclass(kw_only=True)
class ChipConfig(BaseModel):
    num_sa: int
    num_vu: int
    hbm_bw_GBps: float
    hbm_latency_ns: int = 500
    vmem_size_MB: int
    freq_GHz: float

    sa_dim: int = 128

    hbm_size_GB: int
    ici_bw_GBps: float
    dcn_bw_GBps: float

    ici_latency_ns: int = int(5e3)
    dcn_latency_ns: int = int(5e3)

    TDP_W: float
    # idle_power_W: float
    min_power_W: float
    avg_power_W: float
    max_power_W: float
    HBM_GBps_per_W: float
    ICI_GBps_per_W: float
    ICI_topology: str

    embodied_carbon_kgCO2: float

    use_vu_for_small_matmul: bool = True
    '''
    Lower the MatMul ops that are too small to VU to avoid SA padding overhead,
    if using VU results in faster compute time.
    '''

    @property
    def peak_SA_tflops_per_sec(self) -> float:
        '''
        Peak TFLOPs per second of all SAs.
        '''
        return self.num_sa * (self.sa_dim ** 2) * 2 * self.freq_GHz * 1e9 / 1e12

    @property
    def peak_VU_tflops_per_sec(self) -> float:
        '''
        Peak TFLOPs per second of all VUs.
        Assume each VU is 8*128 SIMD ALU.
        '''
        return self.num_vu * (8 * 128) * self.freq_GHz * 1e9 / 1e12

    @property
    def peak_tflops_per_sec(self) -> float:
        '''
        Peak TFLOPs per second of the chip.
        '''
        return self.peak_SA_tflops_per_sec + self.peak_VU_tflops_per_sec

    ### detailed power model
    static_power_W_per_sa: float
    '''Static power of a single SA in Watts'''
    static_power_W_per_vu: float
    '''Static power of a single VU in Watts'''
    static_power_vmem_W: float
    '''Static power of vmem in Watts'''
    static_power_ici_W: float
    '''Static power of ICI in Watts'''
    static_power_hbm_W: float
    '''Static power of HBM controller+PHY in Watts'''
    static_power_other_W: float
    '''Static power of other components in Watts'''

    dynamic_power_W_per_SA: float = 1
    '''Dynamic power of a single SA in Watts'''
    dynamic_power_W_per_VU: float = 1
    '''Dynamic power of a single VU (vector ALU) in Watts'''
    dynamic_power_vmem_W: float = 1
    '''Dynamic power of vmem in Watts'''
    dynamic_power_ici_W_per_GBps: float = 1
    '''Dynamic power of ICI in Watts/GBps'''
    dynamic_power_hbm_W_per_GBps: float = 1
    '''Dynamic power of HBM controller+PHY in Watts/GBps'''
    dynamic_power_other_W: float = 1
    '''Dynamic power of other components in Watts'''

    @property
    def static_power_sa_W(self) -> float:
        return self.static_power_W_per_sa * self.num_sa

    @property
    def static_power_vu_W(self) -> float:
        return self.static_power_W_per_vu * self.num_vu

    @property
    def static_power_vmem_W_per_MB(self) -> float:
        return self.static_power_vmem_W / self.vmem_size_MB

    @property
    def static_power_W(self) -> float:
        return (
            self.static_power_sa_W +
            self.static_power_vu_W +
            self.static_power_vmem_W +
            self.static_power_ici_W +
            self.static_power_hbm_W +
            self.static_power_other_W
        )

    @property
    def idle_power_W(self) -> float:
        '''Currently, assume idle power is the same as static power.'''
        return self.static_power_W

    @property
    def dynamic_power_sa_W(self) -> float:
        return self.dynamic_power_W_per_SA * self.num_sa

    @property
    def dynamic_power_vu_W(self) -> float:
        return self.dynamic_power_W_per_VU * self.num_vu

    @property
    def dynamic_power_hbm_W(self) -> float:
        return self.hbm_bw_GBps * self.dynamic_power_hbm_W_per_GBps

    @property
    def dynamic_power_ici_W(self) -> float:
        return self.ici_bw_GBps * self.dynamic_power_ici_W_per_GBps

    @property
    def dynamic_power_peak_W(self) -> float:
        return (
            self.dynamic_power_sa_W +
            self.dynamic_power_vu_W +
            self.dynamic_power_vmem_W +
            self.dynamic_power_ici_W +
            self.dynamic_power_hbm_W +
            self.dynamic_power_other_W
        )

    @property
    def total_power_peak_W(self) -> float:
        '''
        Total peak power of the chip.
        '''
        return self.static_power_W + self.dynamic_power_peak_W
