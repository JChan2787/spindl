import { describe, it, expect, beforeEach } from "vitest";
import { useCharacterStore } from "./character-store";
import type { CharacterInfo } from "@/types/events";

function createMockCharacter(overrides?: Partial<CharacterInfo>): CharacterInfo {
  return {
    id: "spindle",
    name: "SpindL",
    description: "A test character",
    voice: "kokoro-voice",
    has_avatar: false,
    tags: [],
    ...overrides,
  };
}

describe("useCharacterStore", () => {
  beforeEach(() => {
    useCharacterStore.setState({
      characters: [],
      activeCharacterId: null,
      isLoading: false,
      isSwitchingCharacter: false,
      selectedCharacterId: null,
      selectedCharacterCard: null,
      selectedCharacterHasAvatar: false,
      isLoadingDetail: false,
      avatarCache: {},
      lastAction: {
        type: null,
        characterId: null,
        success: null,
        error: null,
      },
      hasUnsavedChanges: false,
      editedCard: null,
    });
  });

  describe("NANO-077: character switching state", () => {
    it("should initialize isSwitchingCharacter as false", () => {
      expect(useCharacterStore.getState().isSwitchingCharacter).toBe(false);
    });

    it("should set isSwitchingCharacter to true", () => {
      useCharacterStore.getState().setSwitchingCharacter(true);
      expect(useCharacterStore.getState().isSwitchingCharacter).toBe(true);
    });

    it("should set isSwitchingCharacter back to false", () => {
      useCharacterStore.setState({ isSwitchingCharacter: true });
      useCharacterStore.getState().setSwitchingCharacter(false);
      expect(useCharacterStore.getState().isSwitchingCharacter).toBe(false);
    });
  });

  describe("setCharacters", () => {
    it("should set characters and active character", () => {
      const chars = [
        createMockCharacter({ id: "spindle" }),
        createMockCharacter({ id: "mryummers", name: "Mister Yummers" }),
      ];
      useCharacterStore.getState().setCharacters(chars, "spindle");

      const state = useCharacterStore.getState();
      expect(state.characters).toHaveLength(2);
      expect(state.activeCharacterId).toBe("spindle");
    });

    it("should update activeCharacterId on character switch", () => {
      useCharacterStore.setState({ activeCharacterId: "spindle" });
      useCharacterStore.getState().setCharacters([], "mryummers");
      expect(useCharacterStore.getState().activeCharacterId).toBe("mryummers");
    });

    it("should preserve activeCharacterId when active is null", () => {
      useCharacterStore.setState({ activeCharacterId: "spindle" });
      const chars = [createMockCharacter({ id: "spindle" })];
      useCharacterStore.getState().setCharacters(chars, null);
      expect(useCharacterStore.getState().activeCharacterId).toBe("spindle");
    });
  });
});
