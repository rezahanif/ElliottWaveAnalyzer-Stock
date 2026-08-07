<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, computed } from "vue";
import { api, type Asset, type Prediction, type Job } from "../lib/api";
import PredictionChart from "./PredictionChart.vue";
import SignalTensionPanel from "./SignalTensionPanel.vue";

const props = defineProps<{ asset: Asset }>();

const timeframe = ref(props.asset.timeframes[0]?.timeframe ?? "1D");
const predictions = ref<Prediction[]>([]);
const loading = ref(false);
const error = ref<string | null>(null);
const job = ref<Job | null>(null);

const selectedTf = computed(() => props.asset.timeframes.find((t) => t.timeframe === timeframe.value));

async function loadPredictions() {
  loading.value = true;
  error.value = null;
  try {
    const res = await api.getPredictions(props.asset.symbol, timeframe.value);
    predictions.value = res.predictions;
  } catch (e: any) {
    error.value = e.message;
  } finally {
    loading.value = false;
  }
}

async function trigger() {
  error.value = null;
  try {
    const { jobId } = await api.triggerJob(props.asset.id, timeframe.value);
    job.value = await api.getJob(jobId);
  } catch (e: any) {
    error.value = e.message;
  }
}

let unsubscribe: (() => void) | null = null;

onMounted(() => {
  loadPredictions();

  unsubscribe = api.subscribe({
    onPrediction: (p) => {
      if (p.asset === props.asset.symbol && p.timeframe === timeframe.value) {
        loadPredictions();
      }
    },
    onJob: (j) => {
      if (j.asset_id === props.asset.id && j.timeframe === timeframe.value) {
        api.getJob(j.id).then((full) => (job.value = full));
      }
    },
  });
});

onBeforeUnmount(() => unsubscribe?.());

const latest = computed(() => predictions.value[0]);
</script>

<template>
  <div class="tab">
    <div class="controls">
      <select v-model="timeframe" @change="loadPredictions">
        <option v-for="tf in asset.timeframes" :key="tf.timeframe" :value="tf.timeframe">
          {{ tf.timeframe }}
        </option>
      </select>

      <span class="badge" :class="selectedTf?.trained ? 'trained' : 'untrained'">
        {{ selectedTf?.trained ? "trained" : "needs training" }}
      </span>

      <button :disabled="job?.status === 'running'" @click="trigger">
        {{ job?.status === "running" ? "Running…" : "Trigger analysis" }}
      </button>
    </div>

    <p v-if="error" class="error">{{ error }}</p>

    <div v-if="latest" class="summary">
      <div><span class="label">Price</span>{{ latest.current_price ?? "—" }}</div>
      <div><span class="label">Wave</span>{{ latest.wave_position ?? "—" }} ({{ latest.wave_degree ?? "—" }})</div>
      <div><span class="label">Invalidation</span>{{ latest.invalidation_level ?? "—" }}</div>
      <div><span class="label">Cluster</span>{{ latest.cluster_lower ?? "—" }} – {{ latest.cluster_upper ?? "—" }}</div>
    </div>
    <p v-else-if="!loading">No predictions yet for {{ asset.symbol }} / {{ timeframe }}.</p>

    <div class="chart-row">
      <PredictionChart v-if="predictions.length" :predictions="predictions" />
      <SignalTensionPanel :asset="asset.symbol" :timeframe="timeframe" />
    </div>

    <details v-if="job" class="log">
      <summary>Job #{{ job.id }} — {{ job.status }}</summary>
      <pre>{{ job.log_tail }}</pre>
    </details>
  </div>
</template>

<style scoped>
.tab {
  padding: 1rem 0;
}
.chart-row {
  display: flex;
  gap: 1rem;
  align-items: flex-start;
}
.chart-row > :first-child {
  flex: 1;
  min-width: 0;
}
.controls {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  margin-bottom: 1rem;
}
.badge {
  font-size: 0.75rem;
  padding: 2px 8px;
  border-radius: 999px;
}
.badge.trained {
  background: rgba(63, 185, 80, 0.15);
  color: #3fb950;
}
.badge.untrained {
  background: rgba(248, 81, 73, 0.15);
  color: #f85149;
}
.summary {
  display: flex;
  gap: 1.5rem;
  margin-bottom: 1rem;
  font-size: 0.9rem;
}
.label {
  display: block;
  font-size: 0.7rem;
  opacity: 0.6;
}
.error {
  color: #f85149;
}
.log pre {
  max-height: 200px;
  overflow: auto;
  background: rgba(255, 255, 255, 0.03);
  padding: 0.5rem;
  font-size: 0.75rem;
}
</style>
