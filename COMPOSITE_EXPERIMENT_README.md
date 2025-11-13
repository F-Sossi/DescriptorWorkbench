# Composite SIFT Descriptor Pooling Study

## Quick Start

To run the overnight experiment testing 30 SIFT descriptor combinations:

```bash
# From repository root
./run_composite_sift_study.sh

# Or run in background (recommended for overnight):
nohup ./run_composite_sift_study.sh > composite_study.log 2>&1 &

# Monitor progress:
tail -f composite_study.log

# Check if still running:
ps aux | grep experiment_runner
```

## What This Tests

**43 descriptor configurations** testing combinations of:
- Standard SIFT (128D)
- RGBSIFT (384D full color)
- RGBSIFT_CHANNEL_AVG (128D averaged)
- DSPSIFT_V2 (pyramid-aware)
- DSPRGBSIFT_V2 (pyramid-aware color)
- HoNC (Histogram of Normalized Colors, 128D)
- DSPHoNC_V2 (pyramid-aware HoNC)

**5 Aggregation Methods:**
- Simple average
- Weighted average (various ratios)
- Element-wise max
- Element-wise min
- Concatenation (higher dimensionality)

**Dataset:**
- Keypoint set: `sift_2000` (1,164,080 keypoints)
- Scenes: All 116 HPatches scenes
- Images: 696 total images

## Expected Results

**Runtime:** 8-14 hours on modern CPU with parallelization (43 descriptors)

**Hypotheses:**
1. Concatenation will outperform averaging (higher dimensionality helps)
2. Color + grayscale combination will improve on color images
3. DSPSIFT_V2 baseline will be hard to beat (already at 57.25% MAP)
4. Max pooling will capture salient features effectively
5. HoNC (color histogram) may complement SIFT well on color scenes
6. Pyramid-aware pooling (DSPHoNC_V2) will improve on baseline HoNC

## Analyzing Results

After the experiment completes:

### View Top 10 Performers

```bash
cd build
sqlite3 experiments.db
```

```sql
SELECT
    e.descriptor_type,
    printf('%.2f', r.true_map_micro * 100) || '%' as MAP,
    printf('%.2f', r.precision_at_1 * 100) || '%' as P@1
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE e.name = 'composite_sift_pooling_study'
ORDER BY r.true_map_micro DESC
LIMIT 10;
```

### Compare Baselines

```sql
SELECT
    e.descriptor_type,
    printf('%.2f', r.true_map_micro * 100) || '%' as MAP
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE e.name = 'composite_sift_pooling_study'
  AND e.descriptor_type LIKE 'baseline_%'
ORDER BY r.true_map_micro DESC;
```

### Best Aggregation Method

```sql
SELECT
    CASE
        WHEN e.descriptor_type LIKE '%_average' THEN 'average'
        WHEN e.descriptor_type LIKE '%_weighted%' THEN 'weighted_avg'
        WHEN e.descriptor_type LIKE '%_max' THEN 'max'
        WHEN e.descriptor_type LIKE '%_min' THEN 'min'
        WHEN e.descriptor_type LIKE '%_concat' THEN 'concatenate'
        ELSE 'baseline'
    END as method,
    AVG(r.true_map_micro * 100) as avg_map_pct,
    COUNT(*) as count
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE e.name = 'composite_sift_pooling_study'
GROUP BY method
ORDER BY avg_map_pct DESC;
```

### Color vs Grayscale Performance

```sql
-- Performance on color scenes
SELECT
    e.descriptor_type,
    AVG(CASE WHEN hp.scene_type = 'color' THEN r.true_map_micro END) as color_map,
    AVG(CASE WHEN hp.scene_type = 'grayscale' THEN r.true_map_micro END) as gray_map
FROM experiments e
JOIN results r ON e.id = r.experiment_id
WHERE e.name = 'composite_sift_pooling_study'
GROUP BY e.descriptor_type
HAVING color_map IS NOT NULL
ORDER BY color_map DESC
LIMIT 10;
```

## Key Descriptor Combinations to Watch

### Most Promising

1. **sift_rgbsift_avg_concat** (256D)
   - Combines intensity + color, high dimensionality
   - Expected to be top performer

2. **sift_dspsift_v2_concat** (256D)
   - Combines standard + pyramid-aware pooling
   - May beat DSPSIFT_V2 baseline

3. **sift_rgbsift_avg_dspsift_v2_concat** (384D)
   - Triple combination, highest dimensionality
   - Tests if three-way pooling helps

### Most Interesting

4. **sift_rgbsift_avg_max**
   - Element-wise max across grayscale + color
   - Tests if max response helps

5. **sift_rgbsift_avg_weighted_30_70**
   - Favor color over intensity
   - Tests color importance

6. **dspsift_v2_dsprgbsift_v2_average**
   - Pure DSP pooling comparison
   - Tests DSP synergies

## Files

**Experiment Config:** `config/experiments/composite_sift_pooling_study.yaml`
**Study Plan:** `research/composite_sift_pooling_study_plan.md` (detailed analysis plan)
**Runner Script:** `run_composite_sift_study.sh`
**This File:** `COMPOSITE_EXPERIMENT_README.md`

## Troubleshooting

**Experiment fails immediately:**
- Check `./experiment_runner --help` works
- Verify `experiments.db` is accessible
- Check YAML syntax: `python3 -c "import yaml; yaml.safe_load(open('config/experiments/composite_sift_pooling_study.yaml'))"`

**Out of memory:**
- Reduce `num_threads` in YAML (currently auto-detect)
- Disable `save_descriptors` and `save_matches` (already disabled)
- Run on machine with more RAM

**Taking too long:**
- Check if parallelization is working: `htop` should show high CPU usage
- Consider running subset first (comment out some descriptors in YAML)
- Check disk space (database grows during experiment)

**Missing keypoints:**
- Verify: `sqlite3 build/experiments.db "SELECT COUNT(*) FROM locked_keypoints WHERE keypoint_set_id=5;"`
- Should show: 1,164,080 keypoints
- If missing, regenerate: `./keypoint_manager generate-detector ../data sift sift_2000 --max-features 2000`

## Next Steps After Completion

1. **Identify Top 3 Combinations**
   - Which ones beat the best baseline?
   - What aggregation methods work best?

2. **Test on 8000-Keypoint Set**
   - Run winning combinations on `sift_verification_keypoints`
   - More keypoints = better evaluation

3. **Analyze Failure Cases**
   - Which scenes do combinations fail on?
   - Are there patterns?

4. **Test Other Descriptor Families**
   - Apply best aggregation methods to CNN descriptors
   - Test HONC combinations

5. **Write Up Results**
   - Document findings
   - Compare to literature
   - Publish analysis

## References

- Composite descriptor implementation: `src/core/descriptor/extractors/CompositeDescriptorExtractor.hpp`
- YAML configuration guide: `config/YAML_CONFIGURATION_GUIDE.md`
- Descriptor development guide: `skills/descriptor-development/SKILL.md`

---

**Good luck with the experiment!**
Run it overnight and analyze the results in the morning.
