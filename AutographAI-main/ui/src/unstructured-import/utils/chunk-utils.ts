/**
 * Example chunking with a naive approach or tiktoken.
 * This is just a placeholder.
 */
export function chunkText(text: string, chunkSize: number): string[] {
    // A naive approach: split by words
    const words = text.split(/\s+/);
    const chunks: string[] = [];
    let buffer: string[] = [];
  
    for (const word of words) {
      // If adding this word would exceed chunkSize, push
      if ((buffer.join(" ").length + word.length) > chunkSize) {
        chunks.push(buffer.join(" "));
        buffer = [];
      }
      buffer.push(word);
    }
    if (buffer.length) {
      chunks.push(buffer.join(" "));
    }
  
    return chunks;
  }
  