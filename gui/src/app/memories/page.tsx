"use client";

import { useEffect, useCallback, useRef, useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Globe, Database, Clock, Search, Trash2 } from "lucide-react";
import { getSocket } from "@/lib/socket";
import { useMemoryStore } from "@/lib/stores";
import {
  CollectionStats,
  MemoryList,
  MemoryAddForm,
  MemoryDetailPanel,
  MemoryCard,
} from "@/components/memories";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { MemoryCollectionType, MemoryDocument } from "@/types/events";

// Virtual tab type — "sessions" maps to loading both flashcards + summaries
type PageTab = "global" | "sessions" | "general";

export default function MemoriesPage() {
  const socket = getSocket();
  const store = useMemoryStore();
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [activeTab, setActiveTab] = useState<PageTab>("global");
  const [sessionFilter, setSessionFilter] = useState<string>("all");

  // Load data on mount
  useEffect(() => {
    socket.emit("request_memory_counts", {});
    socket.emit("request_memories", { collection: "global" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-clear action feedback after 3 seconds
  useEffect(() => {
    if (store.lastAction.success !== null) {
      const timer = setTimeout(store.clearActionResult, 3000);
      return () => clearTimeout(timer);
    }
  }, [store.lastAction.success, store.clearActionResult]);

  // Get memories for current tab
  const getActiveMemories = useCallback((): MemoryDocument[] => {
    switch (store.activeCollection) {
      case "global": return store.globalMemories;
      case "general": return store.generalMemories;
      case "flashcards": return store.flashcardMemories;
      case "summaries": return store.summaryMemories;
      default: return [];
    }
  }, [store.activeCollection, store.globalMemories, store.generalMemories, store.flashcardMemories, store.summaryMemories]);

  // Sessions tab: group flashcards + summaries by session_id
  const sessionGroups = useMemo(() => {
    const groups: Record<string, { flashcards: MemoryDocument[]; summary: MemoryDocument | null }> = {};

    for (const fc of store.flashcardMemories) {
      const sid = fc.metadata?.session_id || "__unassigned__";
      if (!groups[sid]) groups[sid] = { flashcards: [], summary: null };
      groups[sid].flashcards.push(fc);
    }

    for (const sm of store.summaryMemories) {
      const sid = sm.metadata?.session_id || "__unassigned__";
      if (!groups[sid]) groups[sid] = { flashcards: [], summary: null };
      if (!groups[sid].summary) groups[sid].summary = sm;
    }

    // Sort session IDs by most recent timestamp descending
    const sortedIds = Object.keys(groups).sort((a, b) => {
      const getLatest = (sid: string) => {
        const g = groups[sid];
        const timestamps = [
          ...g.flashcards.map((fc) => fc.metadata?.timestamp || ""),
          g.summary?.metadata?.timestamp || "",
        ].filter(Boolean);
        return timestamps.sort().reverse()[0] || "";
      };
      return getLatest(b).localeCompare(getLatest(a));
    });

    return { groups, sortedIds };
  }, [store.flashcardMemories, store.summaryMemories]);

  // Unique session IDs for filter dropdown
  const sessionIds = useMemo(() => sessionGroups.sortedIds, [sessionGroups]);

  // Tab change handler
  const handleTabChange = useCallback((tab: string) => {
    const pageTab = tab as PageTab;
    setActiveTab(pageTab);
    store.cancelEdit();
    store.setSearchMode(false);
    store.setLoadingList(true);

    if (pageTab === "sessions") {
      // Load both flashcards and summaries for the Sessions view
      store.setActiveCollection("flashcards");
      socket.emit("request_memories", { collection: "flashcards" });
      socket.emit("request_memories", { collection: "summaries" });
    } else {
      const collection = pageTab as MemoryCollectionType;
      store.setActiveCollection(collection);
      socket.emit("request_memories", { collection });
    }
  }, [socket, store]);

  // Refresh current collection
  const handleRefresh = useCallback(() => {
    store.setLoadingList(true);
    if (activeTab === "sessions") {
      socket.emit("request_memories", { collection: "flashcards" });
      socket.emit("request_memories", { collection: "summaries" });
    } else {
      socket.emit("request_memories", { collection: store.activeCollection });
    }
    socket.emit("request_memory_counts", {});
  }, [socket, store, activeTab]);

  // Search with debounce
  const handleSearchChange = useCallback((query: string) => {
    store.setSearchQuery(query);
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }
    if (query.trim()) {
      searchTimeoutRef.current = setTimeout(() => {
        store.setSearchMode(true);
        store.setSearching(true);
        socket.emit("search_memories", { query, top_k: 20 });
      }, 400);
    } else {
      store.setSearchMode(false);
    }
  }, [socket, store]);

  // Cleanup search timeout
  useEffect(() => {
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, []);

  // CRUD handlers
  const handleAddMemory = useCallback(() => {
    if (activeTab === "global") {
      socket.emit("add_global_memory", { content: store.editContent });
    } else {
      socket.emit("add_general_memory", { content: store.editContent });
    }
  }, [socket, store, activeTab]);

  const handleEditMemory = useCallback(() => {
    if (store.selectedMemory) {
      if (store.selectedCollection === "global") {
        socket.emit("edit_global_memory", {
          id: store.selectedMemory.id,
          content: store.editContent,
        });
      } else {
        socket.emit("edit_general_memory", {
          id: store.selectedMemory.id,
          content: store.editContent,
        });
      }
    }
  }, [socket, store]);

  const handleDeleteMemory = useCallback(() => {
    if (store.selectedMemory && store.selectedCollection) {
      socket.emit("delete_memory", {
        collection: store.selectedCollection,
        id: store.selectedMemory.id,
      });
      store.selectMemory(null, null);
    }
  }, [socket, store]);

  const handlePromote = useCallback((deleteSource: boolean) => {
    if (store.selectedMemory && store.selectedCollection) {
      socket.emit("promote_memory", {
        source_collection: store.selectedCollection as "flashcards" | "summaries",
        id: store.selectedMemory.id,
        delete_source: deleteSource,
      });
    }
  }, [socket, store]);

  const handleClearFlashcards = useCallback(() => {
    socket.emit("clear_flashcards", {});
  }, [socket]);

  // Determine right panel content
  const isSaving = store.lastAction.type !== null && store.lastAction.success === null;
  const showAddForm = store.isNewMemory;
  const showDetail = !showAddForm;

  // Format session ID for display
  const formatSessionId = (sid: string) => {
    if (sid === "__unassigned__") return "Unassigned";
    if (sid.length > 30) return sid.substring(0, 30) + "...";
    return sid;
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Memories</h1>
          <CollectionStats counts={store.counts} enabled={store.memoryEnabled} />
        </div>
        <div className="relative w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Semantic search..."
            className="pl-10"
            value={store.searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            disabled={!store.memoryEnabled}
          />
        </div>
      </div>

      {/* Action feedback toast */}
      {store.lastAction.success !== null && (
        <div
          className={`text-sm px-3 py-2 rounded-lg ${
            store.lastAction.success
              ? "bg-green-500/10 text-green-500 border border-green-500/20"
              : "bg-destructive/10 text-destructive border border-destructive/20"
          }`}
        >
          {store.lastAction.success
            ? `Memory ${store.lastAction.type === "promote" ? "escalated" : store.lastAction.type === "clear" ? "flash cards cleared" : `${store.lastAction.type}ed`} successfully`
            : `Error: ${store.lastAction.error || "Unknown error"}`}
        </div>
      )}

      {/* Search results overlay */}
      {store.isSearchMode ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Card className="lg:col-span-2">
            <CardHeader className="py-3 px-4">
              <CardTitle className="text-base flex items-center gap-2">
                <Search className="h-5 w-5" />
                Search Results
                <span className="text-muted-foreground font-normal text-sm">
                  ({store.searchResults.length} results for &quot;{store.searchQuery}&quot;)
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="px-2 pb-2">
              <ScrollArea className="h-[calc(100vh-320px)]">
                {store.isSearching ? (
                  <div className="flex items-center justify-center py-12">
                    <p className="text-sm text-muted-foreground">Searching...</p>
                  </div>
                ) : store.searchResults.length === 0 ? (
                  <div className="flex items-center justify-center py-12">
                    <p className="text-sm text-muted-foreground">No results found</p>
                  </div>
                ) : (
                  <div className="space-y-1 px-2">
                    {store.searchResults.map((result) => (
                      <MemoryCard
                        key={result.id}
                        memory={result}
                        collection={result.collection}
                        isSelected={store.selectedMemory?.id === result.id}
                        onSelect={() => store.selectMemory(result, result.collection)}
                        distance={result.distance}
                      />
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
          <MemoryDetailPanel
            memory={store.selectedMemory}
            collection={store.selectedCollection}
            isEditing={store.isEditing}
            isNewMemory={store.isNewMemory}
            editContent={store.editContent}
            onEditContent={store.setEditContent}
            onSave={handleEditMemory}
            onStartEdit={() => {
              if (store.selectedMemory) store.startEditMemory(store.selectedMemory);
            }}
            onCancel={store.cancelEdit}
            onDelete={handleDeleteMemory}
            onPromote={handlePromote}
            isSaving={isSaving}
          />
        </div>
      ) : (
        /* Main tabbed content */
        <Tabs value={activeTab} onValueChange={handleTabChange}>
          <TabsList>
            <TabsTrigger value="global" className="gap-1.5">
              <Globe className="h-4 w-4" />
              Global
            </TabsTrigger>
            <TabsTrigger value="sessions" className="gap-1.5">
              <Clock className="h-4 w-4" />
              Sessions
            </TabsTrigger>
            <TabsTrigger value="general" className="gap-1.5">
              <Database className="h-4 w-4" />
              General
            </TabsTrigger>
          </TabsList>

          {/* Global tab — cross-character user-entered memories */}
          <TabsContent value="global">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <MemoryList
                  memories={store.globalMemories}
                  isLoading={store.isLoadingList}
                  collection="global"
                  selectedMemoryId={store.selectedMemory?.id ?? null}
                  onSelect={(memory) => store.selectMemory(memory, "global")}
                  onRefresh={handleRefresh}
                  onCreateNew={store.startNewMemory}
                />
              </div>
              <div>
                {showAddForm && activeTab === "global" ? (
                  <MemoryAddForm
                    content={store.editContent}
                    onContentChange={store.setEditContent}
                    onSave={handleAddMemory}
                    onCancel={store.cancelEdit}
                    isSaving={isSaving}
                    isGlobal
                  />
                ) : showDetail ? (
                  <MemoryDetailPanel
                    memory={store.selectedMemory}
                    collection={store.selectedCollection}
                    isEditing={store.isEditing}
                    isNewMemory={store.isNewMemory}
                    editContent={store.editContent}
                    onEditContent={store.setEditContent}
                    onSave={handleEditMemory}
                    onStartEdit={() => {
                      if (store.selectedMemory) store.startEditMemory(store.selectedMemory);
                    }}
                    onCancel={store.cancelEdit}
                    onDelete={handleDeleteMemory}
                    onPromote={handlePromote}
                    isSaving={isSaving}
                  />
                ) : null}
              </div>
            </div>
          </TabsContent>

          {/* Sessions tab — flashcards + summaries grouped by session */}
          <TabsContent value="sessions">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <Card className="lg:col-span-2">
                <CardHeader className="flex flex-row items-center justify-between py-3 px-4">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Clock className="h-5 w-5" />
                    Session Memories
                    <span className="text-muted-foreground font-normal text-sm">
                      ({store.flashcardMemories.length + store.summaryMemories.length})
                    </span>
                  </CardTitle>
                  <div className="flex items-center gap-2">
                    {sessionIds.length > 1 && (
                      <Select value={sessionFilter} onValueChange={setSessionFilter}>
                        <SelectTrigger className="w-48 h-8 text-xs">
                          <SelectValue placeholder="Filter by session" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">All Sessions</SelectItem>
                          {sessionIds.map((sid) => (
                            <SelectItem key={sid} value={sid}>
                              {formatSessionId(sid)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                    {store.flashcardMemories.length > 0 && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-8"
                        onClick={handleClearFlashcards}
                      >
                        <Trash2 className="h-4 w-4 mr-1" />
                        Clear All
                      </Button>
                    )}
                  </div>
                </CardHeader>
                <CardContent className="px-2 pb-2">
                  <ScrollArea className="h-[calc(100vh-320px)]">
                    {store.isLoadingList ? (
                      <div className="flex items-center justify-center py-12">
                        <p className="text-sm text-muted-foreground">Loading...</p>
                      </div>
                    ) : sessionIds.length === 0 ? (
                      <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                        <Clock className="h-5 w-5" />
                        <p className="text-sm text-muted-foreground mt-3 max-w-xs">
                          No session memories yet. Flash cards and summaries appear here after conversations.
                        </p>
                      </div>
                    ) : (
                      <div className="space-y-4 px-2">
                        {sessionIds
                          .filter((sid) => sessionFilter === "all" || sid === sessionFilter)
                          .map((sid) => {
                            const group = sessionGroups.groups[sid];
                            return (
                              <div key={sid} className="space-y-1">
                                {/* Session header */}
                                <div className="flex items-center gap-2 px-2 py-1">
                                  <Badge variant="outline" className="text-xs font-mono">
                                    {formatSessionId(sid)}
                                  </Badge>
                                  <span className="text-xs text-muted-foreground">
                                    {group.flashcards.length} {group.flashcards.length === 1 ? "entry" : "entries"}
                                    {group.summary ? " + summary" : ""}
                                  </span>
                                </div>

                                {/* Summary as header card if present */}
                                {group.summary && (
                                  <div
                                    className={`w-full text-left p-3 rounded-lg border transition-colors hover:bg-accent/50 cursor-pointer ${
                                      store.selectedMemory?.id === group.summary.id
                                        ? "border-primary bg-accent"
                                        : "border-blue-500/20 bg-blue-500/5"
                                    }`}
                                    onClick={() => store.selectMemory(group.summary!, "summaries")}
                                  >
                                    <div className="flex items-center gap-2 mb-1">
                                      <Badge variant="outline" className="text-xs px-1.5 py-0 bg-blue-500/10 text-blue-400 border-blue-500/20">
                                        Summary
                                      </Badge>
                                    </div>
                                    <p className="text-sm leading-relaxed text-muted-foreground">
                                      {group.summary.content.length > 200
                                        ? group.summary.content.substring(0, 200) + "..."
                                        : group.summary.content}
                                    </p>
                                  </div>
                                )}

                                {/* Flash card entries */}
                                {group.flashcards.map((fc) => (
                                  <MemoryCard
                                    key={fc.id}
                                    memory={fc}
                                    collection="flashcards"
                                    isSelected={store.selectedMemory?.id === fc.id}
                                    onSelect={() => store.selectMemory(fc, "flashcards")}
                                  />
                                ))}
                              </div>
                            );
                          })}
                      </div>
                    )}
                  </ScrollArea>
                </CardContent>
              </Card>
              <div>
                <MemoryDetailPanel
                  memory={store.selectedMemory}
                  collection={store.selectedCollection}
                  isEditing={store.isEditing}
                  isNewMemory={store.isNewMemory}
                  editContent={store.editContent}
                  onEditContent={store.setEditContent}
                  onSave={handleEditMemory}
                  onStartEdit={() => {
                    if (store.selectedMemory) store.startEditMemory(store.selectedMemory);
                  }}
                  onCancel={store.cancelEdit}
                  onDelete={handleDeleteMemory}
                  onPromote={handlePromote}
                  isSaving={isSaving}
                />
              </div>
            </div>
          </TabsContent>

          {/* General tab — per-character durable memories */}
          <TabsContent value="general">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <MemoryList
                  memories={store.generalMemories}
                  isLoading={store.isLoadingList}
                  collection="general"
                  selectedMemoryId={store.selectedMemory?.id ?? null}
                  onSelect={(memory) => store.selectMemory(memory, "general")}
                  onRefresh={handleRefresh}
                  onCreateNew={store.startNewMemory}
                />
              </div>
              <div>
                {showAddForm && activeTab === "general" ? (
                  <MemoryAddForm
                    content={store.editContent}
                    onContentChange={store.setEditContent}
                    onSave={handleAddMemory}
                    onCancel={store.cancelEdit}
                    isSaving={isSaving}
                  />
                ) : showDetail ? (
                  <MemoryDetailPanel
                    memory={store.selectedMemory}
                    collection={store.selectedCollection}
                    isEditing={store.isEditing}
                    isNewMemory={store.isNewMemory}
                    editContent={store.editContent}
                    onEditContent={store.setEditContent}
                    onSave={handleEditMemory}
                    onStartEdit={() => {
                      if (store.selectedMemory) store.startEditMemory(store.selectedMemory);
                    }}
                    onCancel={store.cancelEdit}
                    onDelete={handleDeleteMemory}
                    onPromote={handlePromote}
                    isSaving={isSaving}
                  />
                ) : null}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
}
