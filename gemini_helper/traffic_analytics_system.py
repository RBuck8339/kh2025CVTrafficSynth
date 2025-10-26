import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import requests

class TrafficAnalyticsCollector:

    def __init__(self, output_dir='traffic_analytics'):
        self.output_dir = output_dir
        self.setup_directories()
        self.analytics_data = {
            'incidents': [],
            'vehicle_counts': defaultdict(int),
            'hourly_traffic': defaultdict(int),
            'intersection_events': [],
            'speed_data': [],
            'weather_conditions': []
        }

    def setup_directories(self):
        dirs = ['raw_data', 'analytics', 'reports', 'visualizations']
        for d in dirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def collect_incident_data(self, carla_world, ego_vehicle, frame_number,
                             timestamp, weather='clear', time_of_day='day'):
        ego_location = ego_vehicle.get_location()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed_kmh = 3.6 * np.sqrt(ego_velocity.x**2 + ego_velocity.y**2 + ego_velocity.z**2)

        incident_data = {
            'frame': frame_number,
            'timestamp': timestamp.isoformat(),
            'ego_speed_kmh': ego_speed_kmh,
            'ego_location': {'x': ego_location.x, 'y': ego_location.y, 'z': ego_location.z},
            'weather': weather,
            'time_of_day': time_of_day,
            'vehicles': [],
            'near_misses': [],
            'critical_events': []
        }

        # Analyze all nearby vehicles
        for vehicle in carla_world.get_actors().filter('vehicle.*'):
            if vehicle.id == ego_vehicle.id:
                continue

            vehicle_location = vehicle.get_location()
            distance = ego_location.distance(vehicle_location)

            if distance < 50.0:  # Only analyze nearby vehicles
                vehicle_velocity = vehicle.get_velocity()
                vehicle_speed_kmh = 3.6 * np.sqrt(
                    vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2
                )

                vehicle_info = {
                    'id': vehicle.id,
                    'type': vehicle.type_id,
                    'distance': distance,
                    'speed_kmh': vehicle_speed_kmh,
                    'location': {
                        'x': vehicle_location.x,
                        'y': vehicle_location.y,
                        'z': vehicle_location.z
                    },
                    'relative_angle': self._calculate_relative_angle(
                        ego_vehicle, vehicle
                    )
                }

                incident_data['vehicles'].append(vehicle_info)

                # Classify incidents by severity
                if distance < 3.0:
                    severity = 'critical'
                    incident_type = 'critical_near_miss'
                elif distance < 5.0:
                    severity = 'high'
                    incident_type = 'near_miss'
                elif distance < 7.0:
                    severity = 'medium'
                    incident_type = 'close_proximity'
                else:
                    severity = 'low'
                    incident_type = 'normal_traffic'

                # Detect specific event types
                event_type = self._classify_event_type(
                    ego_vehicle, vehicle, distance, vehicle_speed_kmh, ego_speed_kmh
                )

                if severity in ['critical', 'high']:
                    near_miss = {
                        'severity': severity,
                        'type': incident_type,
                        'event_type': event_type,
                        'distance': distance,
                        'ego_speed': ego_speed_kmh,
                        'other_speed': vehicle_speed_kmh,
                        'vehicle_type': vehicle.type_id,
                        'location': vehicle_info['location'],
                        'time_of_day': time_of_day,
                        'weather': weather
                    }
                    incident_data['near_misses'].append(near_miss)

                    # Track in analytics
                    self.analytics_data['incidents'].append({
                        **near_miss,
                        'timestamp': timestamp.isoformat(),
                        'frame': frame_number
                    })

                # Count vehicle types
                self.analytics_data['vehicle_counts'][vehicle.type_id] += 1

        # Track hourly patterns
        hour = timestamp.hour
        self.analytics_data['hourly_traffic'][hour] += len(incident_data['vehicles'])

        # Track speed data
        self.analytics_data['speed_data'].append({
            'timestamp': timestamp.isoformat(),
            'speed_kmh': ego_speed_kmh,
            'num_nearby_vehicles': len(incident_data['vehicles'])
        })

        return incident_data

    def _calculate_relative_angle(self, ego_vehicle, other_vehicle):
        """Calculate relative angle between ego and other vehicle"""
        import carla

        ego_transform = ego_vehicle.get_transform()
        other_location = other_vehicle.get_location()
        ego_location = ego_transform.location

        # Vector from ego to other
        dx = other_location.x - ego_location.x
        dy = other_location.y - ego_location.y

        # Ego's forward vector
        ego_yaw = np.radians(ego_transform.rotation.yaw)
        forward_x = np.cos(ego_yaw)
        forward_y = np.sin(ego_yaw)

        # Calculate angle
        angle = np.arctan2(dy, dx) - np.arctan2(forward_y, forward_x)
        angle = np.degrees(angle)

        # Normalize to [-180, 180]
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360

        return angle

    def _classify_event_type(self, ego_vehicle, other_vehicle, distance,
                           other_speed, ego_speed):
        angle = abs(self._calculate_relative_angle(ego_vehicle, other_vehicle))
        speed_diff = abs(ego_speed - other_speed)

        # Front collision risk
        if angle < 30 and distance < 5.0:
            if ego_speed > other_speed + 10:
                return 'rear_end_risk'
            else:
                return 'frontal_approach'

        # Side collision risk
        elif 60 < angle < 120 and distance < 5.0:
            return 'side_collision_risk'

        # Lane change risk
        elif 30 < angle < 60 and distance < 7.0:
            return 'lane_change_conflict'

        # Merging scenario
        elif speed_diff < 5 and distance < 10.0:
            return 'merging_scenario'

        # High speed approach
        elif ego_speed > 60 and distance < 10.0:
            return 'high_speed_approach'

        return 'normal_interaction'

    def save_analytics_session(self, session_name):
        output_path = os.path.join(self.output_dir, 'analytics',
                                   f'{session_name}_analytics.json')

        # Convert defaultdicts to regular dicts for JSON serialization
        analytics_export = {
            'incidents': self.analytics_data['incidents'],
            'vehicle_counts': dict(self.analytics_data['vehicle_counts']),
            'hourly_traffic': dict(self.analytics_data['hourly_traffic']),
            'speed_data': self.analytics_data['speed_data'],
            'summary': self._generate_summary()
        }

        with open(output_path, 'w') as f:
            json.dump(analytics_export, f, indent=2)

        print(f"✓ Analytics saved to {output_path}")
        return output_path


    def _generate_summary(self):
        incidents = self.analytics_data['incidents']

        if not incidents:
            return {'message': 'No incidents recorded'}

        return {
            'total_incidents': len(incidents),
            'critical_count': sum(1 for i in incidents if i['severity'] == 'critical'),
            'high_risk_count': sum(1 for i in incidents if i['severity'] == 'high'),
            'most_common_vehicle': max(self.analytics_data['vehicle_counts'].items(),
                                      key=lambda x: x[1])[0] if self.analytics_data['vehicle_counts'] else None,
            'average_incident_distance': np.mean([i['distance'] for i in incidents]),
            'peak_hour': max(self.analytics_data['hourly_traffic'].items(),
                           key=lambda x: x[1])[0] if self.analytics_data['hourly_traffic'] else None,
            'incident_types': dict(Counter([i['event_type'] for i in incidents]))
        }


class TrafficAnalyticsProcessor:
    def __init__(self, analytics_file):
        with open(analytics_file, 'r') as f:
            self.data = json.load(f)
        self.incidents = pd.DataFrame(self.data['incidents'])

    def generate_comprehensive_report(self, output_dir='reports'):
        os.makedirs(output_dir, exist_ok=True)

        report = {
            'summary': self._analyze_summary(),
            'temporal_analysis': self._analyze_temporal_patterns(),
            'vehicle_analysis': self._analyze_vehicle_patterns(),
            'location_analysis': self._analyze_location_hotspots(),
            'severity_analysis': self._analyze_severity_distribution(),
            'environmental_factors': self._analyze_environmental_factors(),
            'recommendations': []
        }

        # Generate visualizations
        self._create_visualizations(output_dir)

        # Save report
        report_path = os.path.join(output_dir, 'traffic_analytics_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Report generated: {report_path}")
        return report

    def _analyze_summary(self):
        return {
            'total_incidents': len(self.incidents),
            'critical_incidents': len(self.incidents[self.incidents['severity'] == 'critical']),
            'high_risk_incidents': len(self.incidents[self.incidents['severity'] == 'high']),
            'average_distance': self.incidents['distance'].mean(),
            'min_distance': self.incidents['distance'].min(),
            'average_speed': self.incidents['ego_speed'].mean(),
            'max_speed': self.incidents['ego_speed'].max()
        }

    def _analyze_temporal_patterns(self):
        """Analyze incidents by time"""
        self.incidents['timestamp'] = pd.to_datetime(self.incidents['timestamp'])
        self.incidents['hour'] = self.incidents['timestamp'].dt.hour

        hourly_counts = self.incidents.groupby('hour').size().to_dict()
        time_of_day_counts = self.incidents.groupby('time_of_day').size().to_dict()

        return {
            'hourly_distribution': hourly_counts,
            'time_of_day_distribution': time_of_day_counts,
            'peak_hour': max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None,
            'safest_hour': min(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None
        }

    def _analyze_vehicle_patterns(self):
        vehicle_involvement = self.incidents.groupby('vehicle_type').agg({
            'severity': lambda x: (x == 'critical').sum(),
            'distance': 'mean',
            'other_speed': 'mean'
        }).to_dict()

        return {
            'vehicle_type_incidents': self.incidents['vehicle_type'].value_counts().to_dict(),
            'critical_by_vehicle': vehicle_involvement
        }

    def _analyze_location_hotspots(self):
        # Group nearby locations (within 10 units)
        locations = self.incidents[['location']].copy()

        # Simple clustering by rounding coordinates
        self.incidents['loc_x_cluster'] = self.incidents['location'].apply(
            lambda x: round(x['x'] / 10) * 10
        )
        self.incidents['loc_y_cluster'] = self.incidents['location'].apply(
            lambda x: round(x['y'] / 10) * 10
        )

        hotspots = self.incidents.groupby(['loc_x_cluster', 'loc_y_cluster']).size()
        top_hotspots = hotspots.nlargest(5).to_dict()

        return {
            'top_hotspots': [
                {'location': {'x': k[0], 'y': k[1]}, 'incident_count': v}
                for k, v in top_hotspots.items()
            ]
        }

    def _analyze_severity_distribution(self):
        """Analyze severity patterns"""
        severity_by_speed = self.incidents.groupby(
            pd.cut(self.incidents['ego_speed'], bins=[0, 30, 60, 90, 150])
        )['severity'].value_counts().to_dict()

        return {
            'severity_distribution': self.incidents['severity'].value_counts().to_dict(),
            'event_type_distribution': self.incidents['event_type'].value_counts().to_dict(),
            'severity_by_speed_range': {str(k): v for k, v in severity_by_speed.items()}
        }

    def _analyze_environmental_factors(self):
        """Analyze weather and time of day impact"""
        weather_impact = self.incidents.groupby('weather').agg({
            'severity': lambda x: (x == 'critical').sum(),
            'distance': 'mean'
        }).to_dict()

        time_impact = self.incidents.groupby('time_of_day').agg({
            'severity': lambda x: (x == 'critical').sum(),
            'distance': 'mean'
        }).to_dict()

        return {
            'weather_impact': weather_impact,
            'time_of_day_impact': time_impact
        }

    def _create_visualizations(self, output_dir):
        """Create visualization charts"""

        # 1. Incident severity distribution
        plt.figure(figsize=(10, 6))
        self.incidents['severity'].value_counts().plot(kind='bar', color='steelblue')
        plt.title('Incident Severity Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Severity Level')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'severity_distribution.png'), dpi=300)
        plt.close()

        # 2. Hourly incident patterns
        plt.figure(figsize=(12, 6))
        hourly = self.incidents.groupby('hour').size()
        hourly.plot(kind='line', marker='o', linewidth=2, markersize=8, color='coral')
        plt.title('Incidents by Hour of Day', fontsize=16, fontweight='bold')
        plt.xlabel('Hour')
        plt.ylabel('Number of Incidents')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hourly_patterns.png'), dpi=300)
        plt.close()

        # 3. Event type distribution
        plt.figure(figsize=(12, 6))
        self.incidents['event_type'].value_counts().plot(kind='barh', color='mediumseagreen')
        plt.title('Traffic Event Types', fontsize=16, fontweight='bold')
        plt.xlabel('Count')
        plt.ylabel('Event Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'event_types.png'), dpi=300)
        plt.close()

        # 4. Speed vs Distance scatter
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.incidents['ego_speed'],
                            self.incidents['distance'],
                            c=self.incidents['severity'].map({'critical': 2, 'high': 1, 'medium': 0}),
                            cmap='RdYlGn_r', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Severity (2=Critical, 1=High, 0=Medium)')
        plt.title('Speed vs Distance Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Ego Vehicle Speed (km/h)')
        plt.ylabel('Distance to Other Vehicle (m)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speed_distance_analysis.png'), dpi=300)
        plt.close()

        # 5. Weather impact
        plt.figure(figsize=(10, 6))
        weather_severity = pd.crosstab(self.incidents['weather'],
                                      self.incidents['severity'])
        weather_severity.plot(kind='bar', stacked=True, colormap='RdYlGn_r')
        plt.title('Weather Conditions vs Incident Severity', fontsize=16, fontweight='bold')
        plt.xlabel('Weather Condition')
        plt.ylabel('Count')
        plt.legend(title='Severity')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weather_impact.png'), dpi=300)
        plt.close()

        print(f"✓ Visualizations saved to {output_dir}/")


class AISafetyRecommendationEngine:
    """Generate AI-powered safety recommendations using Google Gemini"""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('AIzaSyCHnqaXYQL_ZDPr5pE6mgqgW9cdnsBNMpk')
        if not self.api_key:
            print("⚠️  Warning: No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
            print("Get your key at: https://makersuite.google.com/app/apikey")

    def generate_safety_recommendations(self, analytics_report,
                                       intersection_name="Intersection"):
        """
        Generate comprehensive safety recommendations using Gemini AI

        Parameters:
        - analytics_report: Dictionary containing traffic analytics
        - intersection_name: Name/ID of the intersection
        """

        if not self.api_key:
            return self._generate_rule_based_recommendations(analytics_report)

        prompt = self._create_analysis_prompt(analytics_report, intersection_name)

        try:
            # Call Gemini API
            recommendations = self._call_gemini_api(prompt)
            return {
                'intersection': intersection_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'ai_recommendations': recommendations,
                'data_summary': analytics_report['summary'],
                'source': 'gemini-ai'
            }
        except Exception as e:
            print(f"⚠️  Gemini API error: {e}")
            print("Falling back to rule-based recommendations...")
            return self._generate_rule_based_recommendations(analytics_report)

    def _create_analysis_prompt(self, report, intersection_name):
        """Create detailed prompt for Gemini"""

        summary = report['summary']
        temporal = report['temporal_analysis']
        vehicle = report['vehicle_analysis']
        severity = report['severity_analysis']
        environmental = report['environmental_factors']

        prompt = f"""You are a traffic safety expert analyzing data from {intersection_name}.
Based on the following comprehensive traffic analytics, provide detailed safety recommendations.

TRAFFIC INCIDENT SUMMARY:
- Total incidents recorded: {summary['total_incidents']}
- Critical incidents: {summary['critical_incidents']}
- High-risk incidents: {summary['high_risk_incidents']}
- Average distance in incidents: {summary['average_distance']:.2f} meters
- Minimum recorded distance: {summary['min_distance']:.2f} meters
- Average speed during incidents: {summary['average_speed']:.2f} km/h
- Maximum speed recorded: {summary['max_speed']:.2f} km/h

TEMPORAL PATTERNS:
- Peak incident hour: {temporal.get('peak_hour', 'N/A')}
- Safest hour: {temporal.get('safest_hour', 'N/A')}
- Time of day distribution: {temporal['time_of_day_distribution']}
- Hourly pattern: {temporal['hourly_distribution']}

VEHICLE INVOLVEMENT:
- Most involved vehicle types: {vehicle['vehicle_type_incidents']}

INCIDENT SEVERITY DISTRIBUTION:
- By severity: {severity['severity_distribution']}
- By event type: {severity['event_type_distribution']}

ENVIRONMENTAL FACTORS:
- Weather impact: {environmental.get('weather_impact', 'N/A')}
- Time of day impact: {environmental.get('time_of_day_impact', 'N/A')}

Please provide:
1. **Key Safety Concerns**: 3-5 most critical safety issues identified
2. **Infrastructure Recommendations**: Specific physical changes to the intersection (signals, signs, lane markings, etc.)
3. **Traffic Management**: Policy or operational changes (speed limits, signal timing, restricted turns, etc.)
4. **Technology Solutions**: Smart traffic systems, cameras, sensors, or warning systems
5. **Priority Actions**: Top 3 actions that should be implemented immediately
6. **Long-term Improvements**: Strategic improvements for sustained safety
7. **Cost-Benefit Analysis**: Estimated impact vs implementation difficulty for each recommendation

Format your response as structured JSON with these sections.
"""
        return prompt

    def _call_gemini_api(self, prompt):
        """Call Google Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"

        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 2048,
            }
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()

        # Extract text from response
        if 'candidates' in result and len(result['candidates']) > 0:
            text = result['candidates'][0]['content']['parts'][0]['text']

            # Try to parse as JSON if formatted correctly
            try:
                import re
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                pass

            # Return raw text if not JSON
            return {'recommendations_text': text}

        return {'error': 'No response from Gemini'}

    def _generate_rule_based_recommendations(self, report):
        """Fallback: rule-based recommendations without AI"""

        summary = report['summary']
        temporal = report['temporal_analysis']
        severity = report['severity_analysis']

        recommendations = {
            'key_concerns': [],
            'infrastructure': [],
            'traffic_management': [],
            'technology': [],
            'priority_actions': [],
            'source': 'rule-based'
        }

        # Analyze critical incidents
        if summary['critical_incidents'] > 10:
            recommendations['key_concerns'].append(
                f"HIGH: {summary['critical_incidents']} critical near-miss incidents detected"
            )
            recommendations['priority_actions'].append(
                "Immediate intersection redesign needed"
            )

        # Analyze speed
        if summary['average_speed'] > 60:
            recommendations['key_concerns'].append(
                f"Excessive speeds detected (avg: {summary['average_speed']:.1f} km/h)"
            )
            recommendations['traffic_management'].append(
                "Implement speed reduction measures (speed bumps, reduced speed limit)"
            )

        # Analyze minimum distance
        if summary['min_distance'] < 2.0:
            recommendations['key_concerns'].append(
                f"Extremely close encounters recorded ({summary['min_distance']:.2f}m minimum)"
            )
            recommendations['infrastructure'].append(
                "Install physical barriers or improve lane separation"
            )

        # Analyze peak times
        peak_hour = temporal.get('peak_hour')
        if peak_hour is not None:
            recommendations['traffic_management'].append(
                f"Increase traffic control during peak hour ({peak_hour}:00)"
            )

        # Event type analysis
        event_types = severity.get('event_type_distribution', {})
        if 'rear_end_risk' in event_types and event_types['rear_end_risk'] > 5:
            recommendations['infrastructure'].append(
                "Install advance warning systems for sudden stops"
            )

        if 'side_collision_risk' in event_types and event_types['side_collision_risk'] > 5:
            recommendations['infrastructure'].append(
                "Improve intersection visibility and add warning signals"
            )

        # Technology recommendations
        recommendations['technology'].extend([
            "Install smart traffic cameras for real-time monitoring",
            "Implement vehicle-to-infrastructure (V2I) communication",
            "Deploy AI-powered collision warning systems"
        ])

        # Priority actions
        if not recommendations['priority_actions']:
            recommendations['priority_actions'] = [
                "Conduct detailed intersection audit",
                "Install additional signage and road markings",
                "Increase police presence during peak hours"
            ]

        return recommendations

    def save_recommendations(self, recommendations, output_file):
        """Save recommendations to file"""
        with open(output_file, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"✓ Recommendations saved to {output_file}")

    def generate_human_readable_report(self, recommendations, output_file):
        """Generate formatted text report"""

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRAFFIC SAFETY ANALYSIS & RECOMMENDATIONS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Intersection: {recommendations.get('intersection', 'N/A')}\n")
            f.write(f"Analysis Date: {recommendations.get('analysis_timestamp', 'N/A')}\n")
            f.write(f"Data Source: {recommendations.get('source', 'N/A')}\n\n")

            # Data summary
            if 'data_summary' in recommendations:
                f.write("-" * 80 + "\n")
                f.write("DATA SUMMARY\n")
                f.write("-" * 80 + "\n")
                for key, value in recommendations['data_summary'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # AI Recommendations
            if 'ai_recommendations' in recommendations:
                ai_recs = recommendations['ai_recommendations']

                if isinstance(ai_recs, dict):
                    for section, content in ai_recs.items():
                        f.write("-" * 80 + "\n")
                        f.write(f"{section.upper().replace('_', ' ')}\n")
                        f.write("-" * 80 + "\n")

                        if isinstance(content, list):
                            for item in content:
                                f.write(f"  • {item}\n")
                        else:
                            f.write(f"{content}\n")
                        f.write("\n")
                else:
                    f.write("-" * 80 + "\n")
                    f.write("RECOMMENDATIONS\n")
                    f.write("-" * 80 + "\n")
                    f.write(str(ai_recs) + "\n\n")

            # Rule-based recommendations
            if 'key_concerns' in recommendations:
                sections = [
                    ('KEY SAFETY CONCERNS', 'key_concerns'),
                    ('INFRASTRUCTURE RECOMMENDATIONS', 'infrastructure'),
                    ('TRAFFIC MANAGEMENT', 'traffic_management'),
                    ('TECHNOLOGY SOLUTIONS', 'technology'),
                    ('PRIORITY ACTIONS', 'priority_actions')
                ]

                for title, key in sections:
                    if key in recommendations and recommendations[key]:
                        f.write("-" * 80 + "\n")
                        f.write(f"{title}\n")
                        f.write("-" * 80 + "\n")
                        for item in recommendations[key]:
                            f.write(f"  • {item}\n")
                        f.write("\n")

        print(f"✓ Human-readable report saved to {output_file}")


# ============================================================================
# PART 4: COMPLETE WORKFLOW
# ============================================================================

def traffic_analytics_workflow():
    """Complete traffic analytics workflow"""

    print("=" * 80)
    print("TRAFFIC ANALYTICS & AI SAFETY RECOMMENDATIONS")
    print("=" * 80)

    # Step 1: Process existing analytics data
    print("\n📊 Step 1: Processing traffic analytics...")

    # Example: Load analytics file (replace with your actual file)
    analytics_file = 'traffic_analytics/analytics/session_analytics.json'

    if not os.path.exists(analytics_file):
        print(f"⚠️  Analytics file not found: {analytics_file}")
        print("Please run data collection first or provide correct path.")
        return

    processor = TrafficAnalyticsProcessor(analytics_file)
    report = processor.generate_comprehensive_report(output_dir='reports')

    # Step 2: Generate AI recommendations
    print("\n🤖 Step 2: Generating AI safety recommendations...")

    # Get Gemini API key from environment or input
    api_key = os.getenv('AIzaSyCHnqaXYQL_ZDPr5pE6mgqgW9cdnsBNMpk')
    if not api_key:
        print("\n💡 To use AI recommendations, set GEMINI_API_KEY environment variable")
        print("Get your free API key at: https://makersuite.google.com/app/apikey")
        print("\nUsing rule-based recommendations instead...\n")

    ai_engine = AISafetyRecommendationEngine(api_key=api_key)
    recommendations = ai_engine.generate_safety_recommendations(
        report,
        intersection_name="Main Street & 5th Avenue"
    )

    # Step 3: Save recommendations
    print("\n💾 Step 3: Saving recommendations...")

    ai_engine.save_recommendations(
        recommendations,
        'reports/safety_recommendations.json'
    )

    ai_engine.generate_human_readable_report(
        recommendations,
        'reports/safety_recommendations.txt'
    )

    print("\n" + "=" * 80)
    print("✅ TRAFFIC ANALYTICS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 reports/traffic_analytics_report.json")
    print("  🤖 reports/safety_recommendations.json")
    print("  📄 reports/safety_recommendations.txt")
    print("  📈 reports/*.png (visualizations)")
    print("\n")


# ============================================================================
# PART 5: INTEGRATION WITH CARLA PIPELINE
# ============================================================================

class EnhancedCARLACollector:
    """Enhanced CARLA collector with integrated analytics"""

    def __init__(self, output_dir='enhanced_carla_data'):
        self.output_dir = output_dir
        self.analytics = TrafficAnalyticsCollector(output_dir)
        self.setup_directories()

    def setup_directories(self):
        dirs = ['images', 'metadata', 'analytics', 'reports']
        for d in dirs:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def collect_with_analytics(self, carla_config):
        """
        Collect CARLA data with full analytics tracking

        carla_config: Dictionary with:
            - host: CARLA server host
            - port: CARLA server port
            - num_frames: Number of frames to collect
            - num_vehicles: Number of traffic vehicles
            - weather: Weather condition
            - time_of_day: Time period
        """
        import carla
        import time

        # Connect to CARLA
        client = carla.Client(carla_config.get('host', 'localhost'),
                            carla_config.get('port', 2000))
        client.set_timeout(10.0)
        world = client.get_world()

        print("✓ Connected to CARLA")

        # Set weather
        weather = self._get_weather_preset(carla_config.get('weather', 'clear'))
        world.set_weather(weather)

        # Spawn ego vehicle with camera
        ego_vehicle, camera = self._spawn_ego_with_camera(world)

        # Spawn traffic
        traffic_vehicles = self._spawn_traffic(world,
                                              carla_config.get('num_vehicles', 50))

        print(f"✓ Spawned {len(traffic_vehicles)} traffic vehicles")

        # Data collection loop
        frame_count = 0
        num_frames = carla_config.get('num_frames', 1000)

        image_data = {'frame': None}
        camera.listen(lambda img: self._process_camera_image(img, image_data))

        start_time = datetime.now()

        try:
            while frame_count < num_frames:
                world.tick()

                if image_data['frame'] is None:
                    continue

                # Collect analytics data
                timestamp = datetime.now()
                incident_data = self.analytics.collect_incident_data(
                    world, ego_vehicle, frame_count, timestamp,
                    weather=carla_config.get('weather', 'clear'),
                    time_of_day=carla_config.get('time_of_day', 'day')
                )

                # Save image if there are incidents
                if incident_data['near_misses']:
                    filename = f"frame_{frame_count:06d}"
                    img_path = os.path.join(self.output_dir, 'images',
                                          f"{filename}.jpg")

                    import cv2
                    cv2.imwrite(img_path,
                              cv2.cvtColor(image_data['frame'], cv2.COLOR_RGB2BGR))

                    # Save metadata
                    meta_path = os.path.join(self.output_dir, 'metadata',
                                           f"{filename}.json")
                    with open(meta_path, 'w') as f:
                        json.dump(incident_data, f, indent=2)

                frame_count += 1

                if frame_count % 100 == 0:
                    elapsed = (datetime.now() - start_time).seconds
                    print(f"Progress: {frame_count}/{num_frames} frames | "
                          f"Incidents: {len(self.analytics.analytics_data['incidents'])} | "
                          f"Time: {elapsed}s")

        except KeyboardInterrupt:
            print("\n⚠️  Collection interrupted by user")

        finally:
            # Cleanup
            camera.destroy()
            ego_vehicle.destroy()
            for vehicle in traffic_vehicles:
                vehicle.destroy()

            print("✓ Cleaned up CARLA actors")

        # Save analytics
        session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        analytics_file = self.analytics.save_analytics_session(session_name)

        return analytics_file

    def _get_weather_preset(self, weather_name):
        """Get CARLA weather preset"""
        import carla

        presets = {
            'clear': carla.WeatherParameters.ClearNoon,
            'cloudy': carla.WeatherParameters.CloudyNoon,
            'wet': carla.WeatherParameters.WetNoon,
            'rain': carla.WeatherParameters.HardRainNoon,
            'fog': carla.WeatherParameters.SoftRainSunset,
            'night': carla.WeatherParameters.ClearNight
        }

        return presets.get(weather_name, carla.WeatherParameters.ClearNoon)

    def _spawn_ego_with_camera(self, world):
        """Spawn ego vehicle with camera"""
        import carla

        bp_lib = world.get_blueprint_library()

        # Spawn vehicle
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        spawn_points = world.get_map().get_spawn_points()
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        ego_vehicle.set_autopilot(True)

        # Attach camera
        camera_bp = bp_lib.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_transform = carla.Transform(carla.Location(x=2.5, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_transform,
                                  attach_to=ego_vehicle)

        return ego_vehicle, camera

    def _spawn_traffic(self, world, num_vehicles):
        """Spawn traffic vehicles"""
        import carla
        import random

        bp_lib = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        vehicles = []
        for i in range(min(num_vehicles, len(spawn_points))):
            vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
            if vehicle:
                vehicle.set_autopilot(True)
                vehicles.append(vehicle)

        return vehicles

    def _process_camera_image(self, image, image_data):
        """Process camera image"""
        import numpy as np
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        image_data['frame'] = array[:, :, :3]


# ============================================================================
# EXAMPLE USAGE & MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with examples"""

    print("\n" + "=" * 80)
    print("TRAFFIC ANALYTICS SYSTEM - MAIN MENU")
    print("=" * 80)
    print("\nChoose an option:")
    print("1. Collect data from CARLA with analytics")
    print("2. Process existing analytics data")
    print("3. Generate AI safety recommendations")
    print("4. Complete workflow (all steps)")
    print("5. View example analytics")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == '1':
        print("\n🚗 Starting CARLA data collection with analytics...\n")

        # Configuration
        config = {
            'host': 'localhost',
            'port': 2000,
            'num_frames': 1000,
            'num_vehicles': 50,
            'weather': 'clear',  # Options: clear, cloudy, wet, rain, fog, night
            'time_of_day': 'day'  # Options: day, dusk, night, dawn
        }

        collector = EnhancedCARLACollector(output_dir='enhanced_carla_data')
        analytics_file = collector.collect_with_analytics(config)

        print(f"\n✓ Collection complete! Analytics saved to: {analytics_file}")

    elif choice == '2':
        print("\n📊 Processing analytics data...\n")

        analytics_file = input("Enter analytics file path: ").strip()
        if not analytics_file:
            analytics_file = 'traffic_analytics/analytics/session_analytics.json'

        if os.path.exists(analytics_file):
            processor = TrafficAnalyticsProcessor(analytics_file)
            report = processor.generate_comprehensive_report()
            print("\n✓ Report generated in 'reports/' directory")
        else:
            print(f"❌ File not found: {analytics_file}")

    elif choice == '3':
        print("\n🤖 Generating AI safety recommendations...\n")

        # Check for API key
        api_key = os.getenv('AIzaSyCHnqaXYQL_ZDPr5pE6mgqgW9cdnsBNMpk')
        if not api_key:
            api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
            if not api_key:
                print("No API key provided. Using rule-based recommendations.")

        report_file = input("Enter analytics report path (or press Enter for default): ").strip()
        if not report_file:
            report_file = 'reports/traffic_analytics_report.json'

        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report = json.load(f)

            intersection_name = input("Enter intersection name: ").strip() or "Intersection A"

            ai_engine = AISafetyRecommendationEngine(api_key=api_key)
            recommendations = ai_engine.generate_safety_recommendations(
                report, intersection_name
            )

            ai_engine.save_recommendations(
                recommendations,
                'reports/safety_recommendations.json'
            )
            ai_engine.generate_human_readable_report(
                recommendations,
                'reports/safety_recommendations.txt'
            )

            print("\n✓ Recommendations saved!")
        else:
            print(f"❌ Report file not found: {report_file}")

    elif choice == '4':
        print("\n🔄 Running complete workflow...\n")
        traffic_analytics_workflow()

    elif choice == '5':
        print("\n📋 Example Analytics Summary\n")
        print_example_analytics()

    else:
        print("Invalid choice!")


def print_example_analytics():
    """Print example analytics to demonstrate capabilities"""

    example = {
        "summary": {
            "total_incidents": 347,
            "critical_incidents": 23,
            "high_risk_incidents": 89,
            "average_distance": 4.2,
            "min_distance": 1.8,
            "average_speed": 52.3,
            "peak_hour": 17
        },
        "top_concerns": [
            "High number of rear-end risks during rush hour",
            "Side collision risks at intersection corners",
            "Excessive speeds during low-traffic periods",
            "Poor visibility during rain conditions"
        ],
        "recommendations": [
            "Install advanced warning signals",
            "Add speed enforcement cameras",
            "Improve lane markings and signage",
            "Consider roundabout conversion",
            "Implement smart traffic signal timing"
        ]
    }

    print(json.dumps(example, indent=2))
    print("\n")
    print("This is example data. Run the full pipeline to get real analytics!")


if __name__ == "__main__":
    # Example: Quick start with default settings
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║  TRAFFIC ANALYTICS & AI SAFETY RECOMMENDATIONS SYSTEM              ║
    ║                                                                    ║
    ║  Features:                                                         ║
    ║  • Collect detailed traffic data from CARLA simulator              ║
    ║  • Track near-misses, vehicle types, speeds, and patterns          ║
    ║  • Generate comprehensive analytics reports                        ║
    ║  • Create data visualizations                                      ║
    ║  • Get AI-powered safety recommendations from Gemini               ║
    ║                                                                    ║
    ║  Quick Start:                                                      ║
    ║  1. Set GEMINI_API_KEY environment variable (optional)             ║
    ║  2. Start CARLA simulator                                          ║
    ║  3. Run this script and choose option 1 or 4                       ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)

    main()
