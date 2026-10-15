# Format specification (draft)

Nothing here is final.

## Container / header

## Color transforms

## Prediction

## Entropy coding

## Alpha channel

- Constant alpha: header flag + value, no payload.
- Binary alpha: see [alpha-binary.md](alpha-binary.md).
- Proposed: alpha is coded after the YCoCg channels, so its model can use the decoded colour
  (including a colour-key mode with no alpha payload). See alpha-binary.md §6.

## Metadata
