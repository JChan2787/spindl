"use client"

import * as React from "react"
import {
  Command,
  CommandEmpty,
  CommandInput,
  CommandItem,
  CommandList,
  CommandLoading,
} from "cmdk"
import { Check, ChevronsUpDown, Loader2, RefreshCw } from "lucide-react"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export interface ModelOption {
  id: string
  name: string
  context_length: number | null
  pricing: { prompt: string; completion: string } | null
}

interface ModelComboboxProps {
  value: string
  onValueChange: (value: string) => void
  models: ModelOption[]
  isLoading: boolean
  error: string | null
  onRefresh: () => void
  placeholder?: string
  disabled?: boolean
}

function formatContextLength(ctx: number | null): string {
  if (ctx === null) return ""
  if (ctx >= 1_000_000) return `${(ctx / 1_000_000).toFixed(1)}M`
  if (ctx >= 1_000) return `${Math.round(ctx / 1_000)}K`
  return String(ctx)
}

function filterModels(models: ModelOption[], search: string): ModelOption[] {
  if (!search) return models
  const lower = search.toLowerCase()
  return models.filter(
    (m) =>
      m.id.toLowerCase().includes(lower) ||
      m.name.toLowerCase().includes(lower)
  )
}

export function ModelCombobox({
  value,
  onValueChange,
  models,
  isLoading,
  error,
  onRefresh,
  placeholder = "Select a model...",
  disabled = false,
}: ModelComboboxProps) {
  const [open, setOpen] = React.useState(false)
  const [search, setSearch] = React.useState("")

  const selectedModel = models.find((m) => m.id === value)
  const displayValue = selectedModel?.name || value || ""
  const filtered = filterModels(models, search)
  const searchMatchesExact = filtered.some((m) => m.id === search)

  return (
    <div className="flex gap-1.5 min-w-0">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="min-w-0 flex-1 justify-between font-normal h-9"
            disabled={disabled}
          >
            <span className="truncate">
              {displayValue || placeholder}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="p-0" align="start">
          <Command shouldFilter={false}>
            <CommandInput
              placeholder="Search models..."
              value={search}
              onValueChange={setSearch}
              className="h-9 border-b border-border bg-transparent px-3 text-sm outline-none placeholder:text-muted-foreground"
            />
            <CommandList className="max-h-60 overflow-y-auto p-1">
              {isLoading && (
                <CommandLoading>
                  <div className="flex items-center justify-center py-6 text-sm text-muted-foreground">
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Fetching models...
                  </div>
                </CommandLoading>
              )}

              {error && !isLoading && (
                <div className="px-2 py-3 text-sm text-destructive">
                  {error}
                </div>
              )}

              {!isLoading && !error && filtered.length === 0 && !search && (
                <CommandEmpty className="py-6 text-center text-sm text-muted-foreground">
                  No models available.
                </CommandEmpty>
              )}

              {!isLoading && !error && filtered.length === 0 && search && (
                <CommandEmpty className="py-6 text-center text-sm text-muted-foreground">
                  No models found.
                </CommandEmpty>
              )}

              {!isLoading &&
                filtered.map((model) => (
                  <CommandItem
                    key={model.id}
                    value={model.id}
                    onSelect={(selectedId) => {
                      onValueChange(selectedId)
                      setOpen(false)
                      setSearch("")
                    }}
                    className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-pointer aria-selected:bg-accent aria-selected:text-accent-foreground"
                  >
                    <Check
                      className={cn(
                        "h-4 w-4 shrink-0",
                        value === model.id ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="truncate font-medium">{model.name}</span>
                      <span className="text-xs text-muted-foreground truncate">
                        {model.id}
                        {model.context_length
                          ? ` · ${formatContextLength(model.context_length)} ctx`
                          : ""}
                      </span>
                    </div>
                  </CommandItem>
                ))}

              {search && !searchMatchesExact && !isLoading && (
                <>
                  <div className="border-t border-border mx-1 my-1" />
                  <CommandItem
                    value={`custom:${search}`}
                    onSelect={() => {
                      onValueChange(search)
                      setOpen(false)
                      setSearch("")
                    }}
                    className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-pointer aria-selected:bg-accent aria-selected:text-accent-foreground"
                  >
                    <span className="text-muted-foreground">Use:</span>
                    <span className="font-mono text-xs">{search}</span>
                  </CommandItem>
                </>
              )}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      <Button
        variant="outline"
        size="icon"
        onClick={onRefresh}
        disabled={isLoading || disabled}
        title="Refresh model list"
        className="shrink-0"
      >
        <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
      </Button>
    </div>
  )
}
