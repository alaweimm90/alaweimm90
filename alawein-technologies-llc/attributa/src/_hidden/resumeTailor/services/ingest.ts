export async function parseResumeFromText(text: string) {
  return { rawText: text, sections: [], normalizedText: text.trim() };
}

export async function parseJDFromText(text: string) {
  return { rawText: text, normalizedText: text.trim() };
}
