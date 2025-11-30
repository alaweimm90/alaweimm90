<script setup lang="ts">
interface Props {
  title?: string;
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
});
</script>

<template>
  <div class="{{COMPONENT_NAME}}">
    <h2 v-if="props.title">{{ props.title }}</h2>
    <slot />
  </div>
</template>

<style scoped>
.{{COMPONENT_NAME}} {
  padding: 1rem;
}
</style>
