<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch } from "vue";
import { createChart, type IChartApi, type UTCTimestamp, LineSeries, AreaSeries } from "lightweight-charts";
import type { Prediction } from "../lib/api";

const props = defineProps<{ predictions: Prediction[] }>();

const container = ref<HTMLDivElement | null>(null);
let chart: IChartApi | null = null;

function toTime(iso: string): UTCTimestamp {
  return Math.floor(new Date(iso).getTime() / 1000) as UTCTimestamp;
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

  draw(props.predictions);
}

function draw(predictions: Prediction[]) {
  if (!chart) return;

  // Sort ascending by time — the API returns newest-first.
  const sorted = [...predictions].sort(
    (a, b) => new Date(a.generated_at).getTime() - new Date(b.generated_at).getTime()
  );
  if (sorted.length === 0) return;

  const currentPriceSeries = chart.addSeries(LineSeries, { color: "#58a6ff", lineWidth: 2, title: "Price" });
  currentPriceSeries.setData(
    sorted
      .filter((p) => p.current_price != null)
      .map((p) => ({ time: toTime(p.generated_at), value: p.current_price as number }))
  );

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

  const bandSeries = chart.addSeries(AreaSeries, {
    topColor: "rgba(227,179,65,0.15)",
    bottomColor: "rgba(227,179,65,0.02)",
    lineColor: "rgba(227,179,65,0.3)",
    lineWidth: 1,
    title: "q90 (7d)",
  });
  bandSeries.setData(
    sorted
      .filter((p) => p.q90_7d != null)
      .map((p) => ({ time: toTime(p.generated_at), value: Number(p.q90_7d) }))
  );

  const latest = sorted[sorted.length - 1];
  if (latest.invalidation_level != null) {
    currentPriceSeries.createPriceLine({
      price: Number(latest.invalidation_level),
      color: "#f85149",
      lineWidth: 1,
      lineStyle: 3,
      title: "invalidation",
    });
  }
  if (latest.cluster_upper != null && latest.cluster_lower != null) {
    currentPriceSeries.createPriceLine({
      price: Number(latest.cluster_upper),
      color: "#3fb950",
      lineWidth: 1,
      lineStyle: 1,
      title: "cluster upper",
    });
    currentPriceSeries.createPriceLine({
      price: Number(latest.cluster_lower),
      color: "#3fb950",
      lineWidth: 1,
      lineStyle: 1,
      title: "cluster lower",
    });
  }

  chart.timeScale().fitContent();
}

onMounted(buildChart);

onBeforeUnmount(() => {
  chart?.remove();
  chart = null;
});

watch(
  () => props.predictions,
  () => {
    // lightweight-charts v5 series are cheap to recreate; simplest correct
    // approach for a low-frequency dashboard is to rebuild the chart wholesale
    // rather than diff series-by-series.
    chart?.remove();
    buildChart();
  },
  { deep: false }
);
</script>

<template>
  <div ref="container" class="chart" />
</template>

<style scoped>
.chart {
  width: 100%;
  height: 420px;
}
</style>
