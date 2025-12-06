# Twitch + YouTube Live Setup Guide

## Overview
Stream interviews and events live on Twitch, simultaneously broadcast to YouTube, then upload edited version to YouTube channel.

---

## Hardware Setup ($300-500)

### Minimum Setup
- **Laptop/PC:** MacBook Pro M1+ or Windows with decent CPU
- **Camera:** Logitech C920 HD Webcam ($50)
- **Microphone:** Audio-Technica AT2020 USB ($99) OR Rode Wireless GO II ($300)
- **Lighting:** 2x Ring Lights ($80)
- **Background:** Backdrop stand + material ($30)
- **Monitor:** For alerts/chat (optional, repurpose old screen)

### Stream PC Suggested Specs
- **CPU:** Intel i5-10400 / AMD Ryzen 5 5600X (or better)
- **RAM:** 16GB minimum
- **GPU:** RTX 3070 or better (for encoding)
- **Internet:** 10+ Mbps upload speed (test at speedtest.net)

---

## Software Stack

### Streaming Engine
**OBS Studio (Open Broadcaster Software)** - FREE

Install from: obsproject.com

**Why OBS?**
- Free and powerful
- Multi-stream to Twitch + YouTube simultaneously
- Professional-level control
- Huge community support

### OBS Setup for Twitch + YouTube Restream

#### Step 1: Install OBS
1. Download from obsproject.com
2. Install and open
3. Go to Settings → Stream

#### Step 2: Connect to Twitch
1. **Settings → Stream**
2. Service: Select "Twitch - Custom Ingest"
3. Server: Closest to you (e.g., US West)
4. Click "Connect Account" → authorize Twitch
5. Server URL auto-populates
6. Click "Get Stream Key" → copy it

#### Step 3: Setup Restream to YouTube
Use **Restream.io** (free plan available):

1. Go to restream.io
2. Sign up (free account)
3. Connect Twitch (already done)
4. Add YouTube as output channel
5. Authorize YouTube account
6. Start streaming on Restream → auto-sends to Twitch + YouTube

**OR** Use OBS built-in custom RTMP:

1. **Settings → Stream**
2. Click "Add" under Multiple Outputs
3. Add YouTube Custom Server RTMP
4. Get YouTube stream key: youtube.com/studio/streams

### Scene Setup in OBS

Create 4 scenes for different stream types:

**Scene 1: Interview Setup**
- Camera (main guest)
- Second camera or frame (host, if two cameras available)
- Overlay (Live It Iconic logo)
- Chat visible on second monitor

**Scene 2: Gameplay/Vlog**
- Screen capture (if showing content)
- Webcam in corner
- Overlay

**Scene 3: Intro Screen**
- Static intro slide
- Music
- Awaiting start message

**Scene 4: Break/Intermission**
- Brand slide
- Music
- "We'll be back in 5 minutes"

### Audio Setup in OBS
1. **Settings → Audio**
2. Sample rate: 48kHz
3. Channels: Stereo
4. Desktop Audio: OFF (unless demoing something)
5. Mic/Aux: Your microphone
6. **In Stream → Audio Track Settings**
   - Track 1: Both (mic + desktop)
   - Make sure audio is coming through

**Audio Levels:**
- Keep peak around -6dB to -3dB
- Absolute max -0dB
- Use noise gate if needed

---

## Stream Settings & Quality

### Video Encoder Settings
**For Twitch (H.264):**
- Resolution: 1920x1080
- FPS: 60 (or 30 if CPU struggles)
- Bitrate: 6,000-8,500 kbps (Twitch recommends 8,500 max)
- CPU Preset: veryfast or fast
- Profile: Main or High

**For YouTube (H.264):**
- Resolution: 1920x1080
- FPS: 60
- Bitrate: 8,000-12,000 kbps
- Same CPU settings

### Key Quality Settings
| Setting | Value |
|---------|-------|
| Resolution | 1920x1080 |
| Frame Rate | 60 fps |
| Bitrate (Twitch) | 6,500 kbps |
| Bitrate (YouTube) | 8,500 kbps |
| Encoder | x264 (H.264) |
| CPU Preset | Fast |
| Buffer Size | 12,000 ms |

**Note:** If internet is unstable, drop to 1280x720 @ 60fps or 1920x1080 @ 30fps

---

## Pre-Stream Checklist

### 24 Hours Before
- [ ] Schedule stream on both Twitch and YouTube
- [ ] Create thumbnail (Canva)
- [ ] Write stream title and description
- [ ] Test internet speed (need 10+ Mbps up)
- [ ] Confirm guest availability
- [ ] Send guest the stream link + tech requirements

### 1 Hour Before
- [ ] Test camera, mic, audio levels
- [ ] Test lighting (no shadows on face)
- [ ] Open OBS, load correct scene
- [ ] Load scenes in order
- [ ] Test stream output (start → stop → check video)
- [ ] Close Discord, Slack, and other apps
- [ ] Close email, notifications
- [ ] Disable Windows/Mac notifications

### 15 Minutes Before
- [ ] Guest logged in and ready
- [ ] Background looks clean
- [ ] Have water nearby
- [ ] Chat moderation set up (if Twitch)
- [ ] Start "Going Live" countdown post
- [ ] Take deep breath

---

## During Stream

### Stream Flow (30-60 min example)

**0:00-2:00 | Intro**
- Welcome, thank you for watching
- Introduce guest + quick bio
- Agenda for next 30-40 minutes

**2:00-15:00 | Story/Background**
- Where guest is from
- How they got into the lifestyle
- First car/major moment

**15:00-35:00 | Deep Dive**
- Their philosophy
- Lessons learned
- Mistakes made
- What's next for them

**35:00-45:00 | Q&A + Chat**
- Read chat questions
- Guest answers live

**45:00-50:00 | Brand Integration**
- "Before we go, let me show you..."
- Show Live It Iconic product
- Story behind it
- How to get it (link in description)

**50:00-60:00 | Outro + Call to Action**
- Thank guest
- "Subscribe on YouTube"
- "Follow on Twitch"
- "Buy merch at [link]"
- End with style

### Pro Tips During Stream
- ✅ Talk naturally (don't read)
- ✅ Maintain eye contact with camera
- ✅ Ask follow-up questions (guests love it)
- ✅ Engage with chat ("I see someone asking...")
- ✅ Pause for 1-2 seconds before answering (avoids dead air)
- ✅ Adjust camera if guest's face goes off frame
- ✅ Have water handy

- ❌ Don't read script word-for-word
- ❌ Don't respond to trolls
- ❌ Don't forget to breathe
- ❌ Don't keep camera too close (creepy angle)
- ❌ Don't forget to look at your guest

---

## Post-Stream

### Immediately After (30 min)
- [ ] End stream properly ("Thank you, goodbye")
- [ ] Stop recording
- [ ] Export chat replay (Twitch)
- [ ] Clip best moments (Twitch Clips)
- [ ] Immediately upload edited highlight to Twitch (within 6 hours)

### Same Day
- [ ] Download raw stream file from OBS
- [ ] Create YouTube-specific upload (different thumbnail)
- [ ] Post YouTube upload (edit to remove dead air, bad segments)
- [ ] Create social media posts with clips
- [ ] Post on Instagram (Reels), TikTok, Twitter

### Within 1 Week
- [ ] Full edited version for YouTube
- [ ] Podcast version uploaded to Spotify/Apple Podcasts
- [ ] Blog post or written transcript
- [ ] Thank you message to guest

---

## Stream Schedule

**Recommended:** Same day/time each week

**Example:**
- **Day:** Wednesday or Thursday
- **Time:** 7 PM (evening, people at home)
- **Length:** 45 minutes to 1 hour

**Consistency matters more than frequency.**

Better to do 1 stream per week reliably than 3 streams erratically.

---

## Monetization (YouTube & Twitch)

### Twitch Monetization
**Requirements:** 50 followers + 500 total watch hours (last 30 days)

**Revenue Streams:**
- Ad revenue (Twitch takes 50%)
- Subscriptions ($5-$25/month, you get 50%)
- Bits/Cheers (you get 1-2 cents per bit, viewer pays 100 bits = $1)
- Donations (through 3rd party like Streamlabs)

### YouTube Monetization
**Requirements:** 1,000 subs + 4,000 watch hours

**Revenue Streams:**
- Ad revenue (CPM varies $2-10 usually)
- Super Chat/Super Likes
- Channel memberships (YouTube takes 30%)
- Affiliate links (car products, gear)

**Realistic Timeline:** 2-3 months of consistent streaming

---

## Troubleshooting

### Stream Keeps Disconnecting
- [ ] Lower bitrate (6000 instead of 8500)
- [ ] Switch to ethernet cable (not WiFi)
- [ ] Close background apps
- [ ] Update OBS to latest version
- [ ] Test with just Twitch first (not dual-streaming)

### Audio Issues
- [ ] Check microphone isn't muted
- [ ] Unplug/replug microphone
- [ ] In OBS, verify audio is being captured
- [ ] Lower audio levels if peaking
- [ ] Try different microphone if available

### Video Quality Degraded
- [ ] Check internet speed
- [ ] Lower resolution to 1280x720
- [ ] Lower FPS to 30 if needed
- [ ] Lower bitrate to 4500-5000
- [ ] Close other apps/downloads

### Chat Not Showing
- [ ] Refresh chat
- [ ] Check if chat is hidden in OBS
- [ ] Verify account is moderator/owner
- [ ] Reload OBS

### Guest's Microphone Sounds Bad
- [ ] Ask them to move mic closer
- [ ] Reduce their input volume
- [ ] Check if they're using laptop speakers (tell them to use headphones)
- [ ] Use Discord audio instead (can be better quality)

---

## Tools & Services

| Tool | Purpose | Cost | Notes |
|------|---------|------|-------|
| **OBS Studio** | Stream software | Free | Open source |
| **Restream.io** | Dual-stream | Free/paid | Free works fine |
| **Canva Pro** | Overlays/graphics | $13/mo | Templates available |
| **Streamlabs** | Overlays + donation | Free | Unnecessary if using OBS + YouTube Super Chat |
| **Riverside.fm** | Record better quality | Free/paid | Can record guest audio separately (pro feature) |
| **Audacity** | Audio editing | Free | If recording podcast separately |

---

## Integration with Website

### On Homepage
1. **Live Stream Widget** - Shows if currently streaming
2. **Upcoming Schedule** - Next stream date/time
3. **Past VODs** - Link to YouTube playlist
4. **Subscribe CTA** - "Subscribe to our YouTube"

### On Nav Bar
- **"Watch"** → Links to YouTube channel
- **"Live"** → Shows Twitch status

### Email List
- Send notification 24h before stream
- Send notification 1h before stream
- Follow-up email with VOD link

---

## First Stream Checklist

**1 Week Before:**
- [ ] Decide on guest + time
- [ ] Test all equipment
- [ ] Practice streaming to private YouTube video (unlisted)
- [ ] Tell friend group about stream

**3 Days Before:**
- [ ] Confirm with guest
- [ ] Write stream title + description
- [ ] Design thumbnail
- [ ] Share on social media: "Going live Wednesday"

**Day Of (1 hour before):**
- [ ] Full tech check
- [ ] Bathroom break
- [ ] Dress nicely
- [ ] Hydrate
- [ ] Start stream 5 min early with just intro slide

**During:**
- [ ] Be natural
- [ ] Ask good questions
- [ ] Show product at end
- [ ] Thank guest warmly
- [ ] End on time

---

## Success Metrics

### First Month Goals
- Consistent weekly stream
- 50-200 viewers
- 1-2K YouTube views from VOD
- 1 or 2 new merch sales traceable to stream

### Months 3-6 Goals
- 500+ viewers per stream
- 5-10K YouTube views per VOD
- Partnerships with other streamers
- Sponsorship inquiries

---

## Next Steps

1. **This week:** Set up OBS, test stream to private YouTube
2. **Next week:** Do first test stream (record but don't broadcast)
3. **Week 3:** First live stream with close friend (low pressure)
4. **Ongoing:** Stream 1x per week, same time

*This setup will evolve as you grow. Start simple, add complexity over time.*
