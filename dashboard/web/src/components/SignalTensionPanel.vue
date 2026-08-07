<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import { api, type SignalTensionResponse } from "../lib/api";
import SignalTensionGauge from "./SignalTensionGauge.vue";

const props = defineProps<{ asset: string; timeframe: string }>();

const tension = ref<SignalTensionResponse | null>(null);
const loading = ref(false);

async function load() {
  loading.value = true;
  tension.value = await api.getSignalTension(props.asset, props.timeframe);
  loading.value = false;
}

onMounted(load);
watch(() => [props.asset, props.timeframe], load);

defineExpose({ reload: load });
</script>

<template>
  <div class="panel">
    <h3>Signal tension</h3>

    <SignalTensionGauge
      v-if="tension?.macro"
      label="Macro · wall street"
      accent="#C98A2B"
      :data="tension.macro"
    />
    <SignalTensionGauge
      v-if="tension?.micro"
      label="Micro · behavioral"
      accent="#3FC7C0"
      :data="tension.micro"
    />

    <p v-if="!loading && !tension?.macro && !tension?.micro" class="empty">
      Signal tension isn't available for {{ asset }} / {{ timeframe }} yet — this needs
      the live ZigZag state exposed from the pipeline first.
    </p>
  </div>
</template>

<style scoped>
.panel {
  min-width: 260px;
}
.panel h3 {
  font-family: ui-monospace, monospace;
  font-size: 0.7rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  opacity: 0.5;
  margin-bottom: 14px;
}
.empty {
  font-size: 0.8rem;
  opacity: 0.55;
  line-height: 1.5;
}
</style>
