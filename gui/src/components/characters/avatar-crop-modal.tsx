"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Loader2, ZoomIn } from "lucide-react";
import type { CropSettings } from "@/types/events";

// ============================================
// Constants
// ============================================

const PREVIEW_SIZE = 400;
const OUTPUT_SIZE = 512;
const SCALE_FACTOR = OUTPUT_SIZE / PREVIEW_SIZE;
const MIN_SCALE = 0.5;
const MAX_SCALE = 3.0;
const SCALE_STEP = 0.1;
const CIRCLE_RADIUS = (PREVIEW_SIZE / 2) - 10; // 10px padding from edge

// ============================================
// Props
// ============================================

interface AvatarCropModalProps {
  image: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onConfirm: (croppedData: string, originalData: string, settings: CropSettings) => void;
  initialSettings?: CropSettings;
}

// ============================================
// Component
// ============================================

export function AvatarCropModal({
  image,
  open,
  onOpenChange,
  onConfirm,
  initialSettings,
}: AvatarCropModalProps) {
  // Crop state
  const [scale, setScale] = useState(1.0);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [offsetAtDragStart, setOffsetAtDragStart] = useState({ x: 0, y: 0 });

  // Image loading
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageSize, setImageSize] = useState({ w: 0, h: 0 });

  // Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // Reset state when modal opens/closes or image changes
  useEffect(() => {
    if (open) {
      if (initialSettings) {
        setScale(initialSettings.scale);
        setOffset({ x: initialSettings.offsetX, y: initialSettings.offsetY });
      } else {
        setScale(1.0);
        setOffset({ x: 0, y: 0 });
      }
      setIsDragging(false);
      setImageLoaded(false);
    }
  }, [open, initialSettings]);

  // Load image to get natural dimensions
  useEffect(() => {
    if (!open || !image) return;

    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
      setImageLoaded(true);
    };
    img.src = image;
  }, [open, image]);

  // ============================================
  // Pan gesture handlers
  // ============================================

  const handlePointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
    setOffsetAtDragStart({ ...offset });
    e.currentTarget.setPointerCapture(e.pointerId);
  }, [offset]);

  const handlePointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (!isDragging) return;
    const dx = e.clientX - dragStart.x;
    const dy = e.clientY - dragStart.y;
    setOffset({
      x: offsetAtDragStart.x + dx,
      y: offsetAtDragStart.y + dy,
    });
  }, [isDragging, dragStart, offsetAtDragStart]);

  const handlePointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    setIsDragging(false);
    e.currentTarget.releasePointerCapture(e.pointerId);
  }, []);

  // ============================================
  // Compute CSS transform for image preview
  // ============================================

  const getImageStyle = useCallback((): React.CSSProperties => {
    if (!imageLoaded || imageSize.w === 0) return {};

    // Scale image so its shorter dimension fills PREVIEW_SIZE
    const aspect = imageSize.w / imageSize.h;
    let displayW: number;
    let displayH: number;

    if (aspect >= 1) {
      // Landscape or square: height fills preview
      displayH = PREVIEW_SIZE;
      displayW = PREVIEW_SIZE * aspect;
    } else {
      // Portrait: width fills preview
      displayW = PREVIEW_SIZE;
      displayH = PREVIEW_SIZE / aspect;
    }

    return {
      position: "absolute" as const,
      width: displayW,
      height: displayH,
      maxWidth: "none",
      left: (PREVIEW_SIZE - displayW) / 2,
      top: (PREVIEW_SIZE - displayH) / 2,
      transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
      transformOrigin: "center center",
      pointerEvents: "none" as const,
      userSelect: "none" as const,
    };
  }, [imageLoaded, imageSize, offset, scale]);

  // ============================================
  // Canvas render (on confirm)
  // ============================================

  const handleConfirm = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !imageLoaded) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width = OUTPUT_SIZE;
    canvas.height = OUTPUT_SIZE;
    ctx.clearRect(0, 0, OUTPUT_SIZE, OUTPUT_SIZE);

    // Mirror the CSS transform logic for pixel-perfect alignment
    const aspect = imageSize.w / imageSize.h;
    let displayW: number;
    let displayH: number;

    if (aspect >= 1) {
      displayH = PREVIEW_SIZE;
      displayW = PREVIEW_SIZE * aspect;
    } else {
      displayW = PREVIEW_SIZE;
      displayH = PREVIEW_SIZE / aspect;
    }

    // Center of output canvas
    const cx = OUTPUT_SIZE / 2;
    const cy = OUTPUT_SIZE / 2;

    ctx.save();

    // Translate to canvas center
    ctx.translate(cx, cy);

    // Apply user offset (scaled to output resolution)
    ctx.translate(offset.x * SCALE_FACTOR, offset.y * SCALE_FACTOR);

    // Apply user zoom (scaled to output resolution)
    ctx.scale(scale * SCALE_FACTOR, scale * SCALE_FACTOR);

    // Draw image centered at origin (matching the CSS layout)
    ctx.drawImage(img, -displayW / 2, -displayH / 2, displayW, displayH);

    ctx.restore();

    const croppedData = canvas.toDataURL("image/png");

    onConfirm(croppedData, image, {
      scale,
      offsetX: offset.x,
      offsetY: offset.y,
    });
  }, [imageLoaded, imageSize, offset, scale, image, onConfirm]);

  // ============================================
  // Render
  // ============================================

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>Crop Avatar</DialogTitle>
          <DialogDescription>
            Drag to pan, use the slider to zoom. The circle shows the framing area.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col items-center gap-4">
          {/* Preview container */}
          <div
            className="relative overflow-hidden bg-black cursor-grab active:cursor-grabbing select-none"
            style={{ width: PREVIEW_SIZE, height: PREVIEW_SIZE }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
          >
            {imageLoaded ? (
              <>
                {/* Image with CSS transform */}
                <img
                  src={image}
                  alt="Crop preview"
                  style={getImageStyle()}
                  draggable={false}
                />

                {/* Circular overlay mask */}
                <svg
                  className="absolute inset-0 pointer-events-none"
                  width={PREVIEW_SIZE}
                  height={PREVIEW_SIZE}
                >
                  <defs>
                    <mask id="crop-circle-mask">
                      <rect width={PREVIEW_SIZE} height={PREVIEW_SIZE} fill="white" />
                      <circle
                        cx={PREVIEW_SIZE / 2}
                        cy={PREVIEW_SIZE / 2}
                        r={CIRCLE_RADIUS}
                        fill="black"
                      />
                    </mask>
                  </defs>
                  <rect
                    width={PREVIEW_SIZE}
                    height={PREVIEW_SIZE}
                    fill="rgba(0,0,0,0.6)"
                    mask="url(#crop-circle-mask)"
                  />
                  <circle
                    cx={PREVIEW_SIZE / 2}
                    cy={PREVIEW_SIZE / 2}
                    r={CIRCLE_RADIUS}
                    fill="none"
                    stroke="rgba(255,255,255,0.3)"
                    strokeWidth="1"
                  />
                </svg>
              </>
            ) : (
              <div className="flex items-center justify-center h-full">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            )}
          </div>

          {/* Zoom slider */}
          <div className="flex items-center gap-3 w-full max-w-xs">
            <ZoomIn className="h-4 w-4 text-muted-foreground shrink-0" />
            <input
              type="range"
              min={MIN_SCALE}
              max={MAX_SCALE}
              step={SCALE_STEP}
              value={scale}
              onChange={(e) => setScale(parseFloat(e.target.value))}
              className="flex-1"
            />
            <Label className="text-xs text-muted-foreground w-10 text-right">
              {scale.toFixed(1)}x
            </Label>
          </div>
        </div>

        {/* Hidden canvas for final render */}
        <canvas ref={canvasRef} className="hidden" />

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleConfirm} disabled={!imageLoaded}>
            Confirm
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
