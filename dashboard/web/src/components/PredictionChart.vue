<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from "vue";
import {
  createChart,
  createSeriesMarkers,
  type IChartApi,
  type UTCTimestamp,
  type Time,
  LineSeries,
  BaselineSeries,
  CandlestickSeries,
  type SeriesMarker,
  type BaselineData,
} from "lightweight-charts";
import { api, type Candle, type Pivot, type Prediction } from "../lib/api";

const props = defineProps<{
  asset: string;
  timeframe: string;
  predictions: Prediction[];
}>();

const container = ref<HTMLDivElement | null>(null);
let chart: IChartApi | null = null;

const candles = ref<Candle[]>([]);
const pivots = ref<Pivot[]>([]);
const candlesError = ref<string | null>(null);

function toTime(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
}

async function loadChartData() {
  try {
    const [c, p] = await Promise.all([
      api.getCandles(props.asset, props.timeframe),
      api.getPivots(props.asset, props.timeframe, "all").catch(() => ({ pivots: [] as Pivot[] })),
    ]);
    candles.value = c.candles;
    pivots.value = p.pivots;
    candlesError.value = null;
  } catch (e: any) {
    candles.value = [];
    candlesError.value = e.message;
  }
  chart?.remove();
  buildChart();
}

function buildChart() {
  if (!container.value) return;
  chart = createChart(container.value, {
    autoSize: true,
    layout: { background: { color: "transparent" }, textColor: "#c9d1d9" },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.06)" },
      horzLines: { color: "rgba(255,255,255,0.06)" },
    },
    timeScale: { timeVisible: true },
  });

  // Primary series: real OHLCV candles.
  const candleSeries = chart.addSeries(CandlestickSeries, {
    upColor: "#26a69a",
    downColor: "#ef5350",
    borderVisible: false,
    wickUpColor: "#26a69a",
    wickDownColor: "#ef5350",
  });
  if (candles.value.length) {
    candleSeries.setData(
      candles.value.map((c) => ({ ...c, time: c.time as Time }))
    );
  }

  const sorted = [...props.predictions].sort(
    (a, b) => new Date(a.generated_at).getTime() - new Date(b.generated_at).getTime()
  );

  // q10/q50/q90 fan — 7d horizon only for v1.
  const q50Series = chart.addSeries(LineSeries, {
    color: "#e3b341",
    lineWidth: 2,
    lineStyle: 2, // dashed — forecast, not observed
    title: "q50 (7d)",
  });
  q50Series.setData(
    sorted
      .filter((p) => p.q50_7d != null)
      .map((p) => ({ time: toTime(p.generated_at), value: Number(p.q50_7d) }))
  );

  const latestQ50 = sorted.filter((p) => p.q50_7d != null).map((p) => Number(p.q50_7d)).pop();

  if (latestQ50 != null) {
    // Upper fan: q50 → q90 (BaselineSeries fills between value and base).
    const q90Series = chart.addSeries(BaselineSeries, {
      baseValue: { type: "price", price: latestQ50 },
      topFillColor1: "rgba(227,179,65,0.15)",
      topFillColor2: "rgba(227,179,65,0.02)",
      topLineColor: "rgba(227,179,65,0.3)",
      bottomFillColor1: "rgba(227,179,65,0.05)",
      bottomFillColor2: "rgba(227,179,65,0.05)",
      bottomLineColor: "rgba(227,179,65,0.2)",
      lineWidth: 1,
      title: "q90 (7d)",
    });
    q90Series.setData(
      sorted
        .filter((p) => p.q90_7d != null)
        .map((p) => ({
          time: toTime(p.generated_at),
          value: Number(p.q90_7d),
        })) as BaselineData<Time>[]
    );

    // Lower fan: q10 → q50.
    const q10Series = chart.addSeries(BaselineSeries, {
      baseValue: { type: "price", price: latestQ50 },
      topFillColor1: "rgba(88,166,255,0.05)",
      topFillColor2: "rgba(88,166,255,0.05)",
      topLineColor: "rgba(88,166,255,0.2)",
      bottomFillColor1: "rgba(88,166,255,0.15)",
      bottomFillColor2: "rgba(88,166,255,0.02)",
      bottomLineColor: "rgba(88,166,255,0.3)",
      lineWidth: 1,
      title: "q10 (7d)",
    });
    q10Series.setData(
      sorted
        .filter((p) => p.q10_7d != null)
        .map((p) => ({
          time: toTime(p.generated_at),
          value: Number(p.q10_7d),
        })) as BaselineData<Time>[]
    );
  }

  const latest = sorted[sorted.length - 1];
  if (latest.invalidation_level != null) {
    candleSeries.createPriceLine({
      price: Number(latest.invalidation_level),
      color: "#f85149",
      lineWidth: 1,
      lineStyle: 3,
      title: "invalidation",
    });
  }
  if (latest.cluster_upper != null && latest.cluster_lower != null) {
    const upper = Number(latest.cluster_upper);
    const lower = Number(latest.cluster_lower);
    // Shaded band between the two levels across the candle range —
    // BaselineSeries with base at cluster_lower, constant value at upper.
    if (candles.value.length) {
      const t0 = candles.value[0].time as Time;
      const t1 = candles.value[candles.value.length - 1].time as Time;
      const zone = chart.addSeries(BaselineSeries, {
        baseValue: { type: "price", price: lower },
        topFillColor1: "rgba(63,185,80,0.12)",
        topFillColor2: "rgba(63,185,80,0.12)",
        topLineColor: "rgba(63,185,80,0.0)",
        bottomFillColor1: "rgba(63,185,80,0.12)",
        bottomFillColor2: "rgba(63,185,80,0.12)",
        bottomLineColor: "rgba(63,185,80,0.0)",
      });
      zone.setData([
        { time: t0, value: upper },
        { time: t1, value: upper },
      ]);
    }
    // Keep the labels — useful regardless of the fill.
    candleSeries.createPriceLine({
      price: upper,
      color: "#3fb950",
      lineWidth: 1,
      lineStyle: 1,
      title: "cluster upper",
    });
    candleSeries.createPriceLine({
      price: lower,
      color: "#3fb950",
      lineWidth: 1,
      lineStyle: 1,
      title: "cluster lower",
    });
  }

  // Pivots as markers — macro visually distinct from micro.
  if (pivots.value.length) {
    const markers: SeriesMarker<Time>[] = pivots.value.map((p) => {
      const isMacro = p.layer === "macro";
      const isHigh = p.swing_type === "High";
      return {
        time: Math.floor(p.timestamp_ms / 1000) as Time,
        position: isHigh ? ("aboveBar" as const) : ("belowBar" as const),
        shape: isMacro ? (isHigh ? ("arrowDown" as const) : ("arrowUp" as const)) : ("circle" as const),
        color: isMacro ? "#e3b341" : "rgba(201,209,217,0.35)",
        size: isMacro ? 1.5 : 0.8,
        text: isMacro ? (p.structure_label && p.structure_label !== "UNKNOWN" ? p.structure_label : "M") : "",
      };
    });
    const markersApi = createSeriesMarkers(candleSeries, markers);
    markersApi.setMarkers(markers);
  }

  if (candles.value.length) chart.timeScale().fitContent();
}

onMounted(() => {
  loadChartData();
});

onBeforeUnmount(() => {
  chart?.remove();
  chart = null;
});

watch(
  () => [props.predictions, props.asset, props.timeframe],
  () => {
    chart?.remove();
    loadChartData();
  },
  { deep: false }
);
</script>

<template>
  <div>
    <p v-if="candlesError" class="error">{{ candlesError }}</p>
    <div ref="container" class="chart" />
  </div>
</template>

<style scoped>
.chart {
  width: 100%;
  height: 420px;
}
.error {
  color: #f85149;
  font-size: 0.8rem;
  margin-bottom: 0.5rem;
}
</style>
