"""
Character Editor UI Tests — NANO-038 Phase 3.

Playwright browser tests that verify the character editor's save flow,
button state gating, and edit-save-reload pipeline through the real UI.

Uses single-config fixture (e2e_harness_local_unified) since UI behavior
is config-independent.
"""

import pytest
from playwright.async_api import expect

from .helpers.assertions import assert_event_received
from .helpers.character_utils import restore_character_file, wait_for_state


# --- Helpers ---


async def navigate_to_character_editor(
    harness, character_name: str = "Test Agent"
):
    """
    Navigate to /characters, select a character, wait for editor to load.

    Args:
        harness: E2EHarness instance with page and frontend_url.
        character_name: Name of the character to select in the list.

    Returns:
        Playwright Locator for the character name input.
    """
    page = harness.page

    # Navigate to characters page
    await page.goto(f"{harness.frontend_url}/characters")

    # Wait for character list to render with the target character
    card_heading = page.locator(
        "h3.font-semibold.truncate", has_text=character_name
    )
    await card_heading.wait_for(state="visible", timeout=15000)

    # Click the character card
    await card_heading.click()

    # Wait for editor: name input should be visible and populated
    name_input = page.locator('[data-testid="character-name-input"]')
    await name_input.wait_for(state="visible", timeout=10000)

    # Wait for the input value to match (detail fetch is async)
    await page.wait_for_function(
        f'document.querySelector(\'[data-testid="character-name-input"]\').value === "{character_name}"',
        timeout=10000,
    )

    return name_input


# --- Tests ---


class TestCharacterEditorUI:
    """Playwright UI tests for the character editor — NANO-038 Phase 3."""

    @pytest.mark.asyncio
    async def test_save_button_enabled_when_idle(
        self, e2e_harness_local_unified
    ):
        """
        In IDLE state with unsaved changes, Save button should be enabled.

        1. Navigate to /characters, select Test Agent
        2. Verify Save button disabled (no changes yet)
        3. Modify character name
        4. Verify "Unsaved changes" indicator appears
        5. Verify Save button becomes enabled
        """
        harness = e2e_harness_local_unified
        page = harness.page

        name_input = await navigate_to_character_editor(harness)

        save_btn = page.locator('[data-testid="save-button"]')
        await save_btn.wait_for(state="visible", timeout=5000)

        # Save should be disabled initially (no unsaved changes)
        await expect(save_btn).to_be_disabled()

        # Make a trivial edit
        await name_input.fill("Test Agent Modified")

        # "Unsaved changes" indicator should appear
        unsaved = page.locator('[data-testid="unsaved-changes-indicator"]')
        await unsaved.wait_for(state="visible", timeout=5000)

        # Save button should now be enabled (IDLE + unsaved changes)
        await expect(save_btn).to_be_enabled()

    @pytest.mark.asyncio
    async def test_save_button_disabled_during_processing(
        self, e2e_harness_local_unified, response_timeout
    ):
        """
        Save button disabled + warning banner shown when agent is PROCESSING.

        1. Navigate to editor, make a change (create unsaved changes)
        2. Confirm Save button is enabled (IDLE state)
        3. Send message to trigger PROCESSING state
        4. Verify Save button becomes disabled
        5. Verify warning banner appears
        6. Drain pipeline for clean teardown
        """
        harness = e2e_harness_local_unified
        page = harness.page

        name_input = await navigate_to_character_editor(harness)

        # Make an edit so we have unsaved changes
        await name_input.fill("Test Agent Processing")

        # Confirm unsaved changes indicator is visible
        unsaved = page.locator('[data-testid="unsaved-changes-indicator"]')
        await unsaved.wait_for(state="visible", timeout=5000)

        # Save should be enabled before we start processing
        save_btn = page.locator('[data-testid="save-button"]')
        await expect(save_btn).to_be_enabled()

        # Send message to push agent into PROCESSING
        harness.clear_events()
        await harness.send_message("Tell me a joke.")

        # Wait for agent to enter PROCESSING via Socket.IO
        await wait_for_state(harness, "processing", timeout=10.0)

        # Save button should now be disabled
        await expect(save_btn).to_be_disabled(timeout=5000)

        # Warning banner should be visible
        banner = page.locator('[data-testid="save-disabled-banner"]')
        await expect(banner).to_be_visible(timeout=5000)

        # Let pipeline finish so harness teardown is clean
        await assert_event_received(
            harness,
            "response",
            timeout=response_timeout,
            message="Waiting for pipeline to finish after processing test",
        )

    @pytest.mark.asyncio
    async def test_full_edit_save_reload_flow(
        self, e2e_harness_local_unified, response_timeout
    ):
        """
        Full edit-save-reload: modify name in UI, save, verify reload succeeds.

        1. Navigate to /characters, select Test Agent
        2. Modify name to unique value
        3. Verify "Unsaved changes" appears
        4. Click Save
        5. Wait for save to complete (unsaved indicator disappears)
        6. Verify character list shows updated name
        7. Restore original name in finally block
        """
        harness = e2e_harness_local_unified
        page = harness.page
        original_name = "Test Agent"
        modified_name = "Test Agent NANO038"

        try:
            name_input = await navigate_to_character_editor(harness)

            # Verify starting state
            assert await name_input.input_value() == original_name

            # --- EDIT ---
            await name_input.fill(modified_name)

            # Unsaved changes indicator should appear
            unsaved = page.locator(
                '[data-testid="unsaved-changes-indicator"]'
            )
            await unsaved.wait_for(state="visible", timeout=5000)

            # --- SAVE ---
            save_btn = page.locator('[data-testid="save-button"]')
            await expect(save_btn).to_be_enabled()
            await save_btn.click()

            # --- VERIFY SAVE COMPLETED ---
            # When save completes, the page re-fetches the character detail,
            # which resets hasUnsavedChanges to false. The unsaved indicator
            # disappears, proving the entire chain succeeded:
            #   updateCharacterApi() -> reloadCharacter() ->
            #   fetchCharacterDetail() -> setCharacterDetail()
            await unsaved.wait_for(state="hidden", timeout=30000)

            # Save button should be disabled again (no unsaved changes)
            await expect(save_btn).to_be_disabled(timeout=5000)

            # Character list should reflect the new name
            list_heading = page.locator(
                "h3.font-semibold.truncate", has_text=modified_name
            )
            await list_heading.wait_for(state="visible", timeout=10000)

        finally:
            # --- RESTORE ---
            # Edit name back to original and save again
            try:
                name_input = page.locator(
                    '[data-testid="character-name-input"]'
                )
                await name_input.fill(original_name)

                unsaved = page.locator(
                    '[data-testid="unsaved-changes-indicator"]'
                )
                await unsaved.wait_for(state="visible", timeout=5000)

                save_btn = page.locator('[data-testid="save-button"]')
                await save_btn.click()

                # Wait for restore to complete
                await unsaved.wait_for(state="hidden", timeout=30000)
            except Exception as restore_error:
                print(
                    f"[WARN] UI restore failed, falling back to file restore: "
                    f"{restore_error}"
                )
                # Fallback: restore the character file directly on disk
                await restore_character_file(harness, "test_agent")
