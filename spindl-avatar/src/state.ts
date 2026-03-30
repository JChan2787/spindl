export type Mode = 'idle' | 'thinking' | 'speaking';

/** Classifier bucket moods + neutral default. */
export type Mood =
  | 'default' | 'amused' | 'melancholy' | 'annoyed' | 'curious'
  | null;

export type ToolMood = 'search' | 'execute' | 'memory' | 'agent' | null;

export interface AvatarState {
  mode: Mode;
  speaking: boolean;
  amplitude: number;
  mood: Mood;
  toolMood: ToolMood;
}

export function createState(): AvatarState {
  return {
    mode: 'idle',
    speaking: false,
    amplitude: 0,
    mood: null,
    toolMood: null,
  };
}
