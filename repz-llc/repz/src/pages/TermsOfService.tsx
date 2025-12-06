import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from "@/ui/atoms/Button";
import { Card, CardContent, CardHeader, CardTitle } from "@/ui/molecules/Card";
import { ArrowLeft, AlertTriangle, Mail, Phone } from 'lucide-react';
import { RepzLogo } from '@/ui/organisms/RepzLogo';

export default function TermsOfService() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-slate-900 to-black">
      {/* Header */}
      <div className="bg-white/95 backdrop-blur-sm shadow-lg border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-4 flex flex-col sm:flex-row items-center justify-between gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => navigate('/')}
            className="text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-all duration-200"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to REPZ
          </Button>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-orange-500 to-red-500 rounded-lg flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-sm">R</span>
            </div>
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-orange-500 to-red-500 font-bold text-2xl tracking-wider">REPZ</span>
          </div>
          <div className="text-center sm:text-right">
            <p className="text-sm text-slate-600 font-medium">Last updated</p>
            <p className="text-xs text-slate-500">February 4, 2025</p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 lg:py-12">
        <Card className="shadow-2xl border-0 bg-white/95 backdrop-blur-sm">
          <CardHeader className="bg-gradient-to-r from-orange-50 via-red-50 to-orange-50 border-b border-gray-200">
            <div className="text-center py-8">
              <CardTitle className="text-4xl md:text-5xl font-bold text-slate-900 mb-4 flex items-center justify-center gap-4">
                <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl flex items-center justify-center shadow-lg">
                  <AlertTriangle className="h-6 w-6 text-white" />
                </div>
                Terms of Service
              </CardTitle>
              <p className="text-slate-600 text-lg max-w-2xl mx-auto leading-relaxed">
                Legal agreement governing your use of the REPZ fitness coaching platform and services
              </p>
              <div className="flex items-center justify-center gap-2 mt-4 text-sm text-slate-500">
                <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                <span>Effective February 4, 2025</span>
              </div>
            </div>
          </CardHeader>
          
          <CardContent className="prose prose-slate max-w-none p-8 lg:p-12">
            <div className="space-y-12">
              
              {/* Table of Contents */}
              <section className="bg-slate-50 border border-slate-200 rounded-xl p-6">
                <h3 className="text-xl font-semibold text-slate-900 mb-4">Table of Contents</h3>
                <div className="grid md:grid-cols-2 gap-2 text-sm">
                  <a href="#agreement" className="text-orange-600 hover:text-orange-700 transition-colors">1. Agreement to Terms</a>
                  <a href="#services" className="text-orange-600 hover:text-orange-700 transition-colors">2. Service Description</a>
                  <a href="#eligibility" className="text-orange-600 hover:text-orange-700 transition-colors">3. Eligibility Requirements</a>
                  <a href="#responsibilities" className="text-orange-600 hover:text-orange-700 transition-colors">4. User Responsibilities</a>
                  <a href="#health" className="text-orange-600 hover:text-orange-700 transition-colors">5. Health & Safety</a>
                  <a href="#payment" className="text-orange-600 hover:text-orange-700 transition-colors">6. Payment & Subscriptions</a>
                  <a href="#privacy" className="text-orange-600 hover:text-orange-700 transition-colors">7. Privacy & Data Protection</a>
                  <a href="#intellectual" className="text-orange-600 hover:text-orange-700 transition-colors">8. Intellectual Property</a>
                  <a href="#liability" className="text-orange-600 hover:text-orange-700 transition-colors">9. Limitation of Liability</a>
                  <a href="#termination" className="text-orange-600 hover:text-orange-700 transition-colors">10. Account Termination</a>
                  <a href="#disputes" className="text-orange-600 hover:text-orange-700 transition-colors">11. Dispute Resolution</a>
                  <a href="#changes" className="text-orange-600 hover:text-orange-700 transition-colors">12. Changes to Terms</a>
                </div>
              </section>

              {/* Introduction */}
              <section id="agreement">
                <div className="flex items-start gap-4 mb-6">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-lg flex items-center justify-center font-bold text-sm">1</div>
                  <h2 className="text-3xl font-semibold text-slate-900">Agreement to Terms</h2>
                </div>
                <div className="bg-orange-50 border-l-4 border-orange-500 p-6 rounded-r-lg mb-6">
                  <p className="text-orange-900 font-medium">
                    By accessing, browsing, or using REPZ in any manner, you acknowledge that you have read, understood, and agree to be bound by these Terms of Service and all applicable laws and regulations.
                  </p>
                </div>
                <p className="text-slate-700 leading-relaxed mb-4">
                  These Terms of Service ("Terms") constitute a legally binding agreement between you ("User", "you", or "your") and REPZ Coach Pro AI, LLC ("REPZ", "we", "us", or "our") governing your use of our fitness coaching platform, mobile applications, and related services (collectively, the "Platform").
                </p>
                <p className="text-slate-700 leading-relaxed mb-4">
                  <strong>Important:</strong> If you do not agree with any part of these Terms, you must immediately discontinue use of our Platform. Continued use constitutes acceptance of these Terms as they may be modified from time to time.
                </p>
                <p className="text-slate-700 leading-relaxed">
                  REPZ is a comprehensive fitness coaching platform utilizing artificial intelligence and certified personal trainers to provide personalized training programs, nutrition guidance, biomarker analysis, and wellness coaching services. Our services are designed for general fitness and wellness improvement purposes only.
                </p>
              </section>

              {/* Service Description */}
              <section id="services">
                <div className="flex items-start gap-4 mb-6">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-lg flex items-center justify-center font-bold text-sm">2</div>
                  <h2 className="text-3xl font-semibold text-slate-900">Service Description</h2>
                </div>
                <p className="text-slate-700 leading-relaxed mb-6">
                  REPZ offers a comprehensive suite of AI-powered fitness coaching services across four distinct subscription tiers:
                </p>
                
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-5">
                    <h4 className="font-semibold text-blue-900 mb-3">Core Platform Services</h4>
                    <ul className="space-y-2 text-blue-800 text-sm">
                      <li>• Personalized training program design</li>
                      <li>• Nutrition planning and macro guidance</li>
                      <li>• Progress tracking and analytics</li>
                      <li>• Exercise form library and instruction</li>
                      <li>• Community support features</li>
                    </ul>
                  </div>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-5">
                    <h4 className="font-semibold text-green-900 mb-3">Advanced Features</h4>
                    <ul className="space-y-2 text-green-800 text-sm">
                      <li>• AI-powered form analysis and feedback</li>
                      <li>• Biomarker integration and health tracking</li>
                      <li>• Performance enhancement protocols</li>
                      <li>• In-person training sessions (select tiers)</li>
                      <li>• Priority coaching support</li>
                    </ul>
                  </div>
                </div>

                <div className="bg-red-50 border-l-4 border-red-500 p-6 rounded-r-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-6 w-6 text-red-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-red-900 mb-2">Medical Disclaimer</h4>
                      <p className="text-red-800 leading-relaxed">
                        <strong>REPZ services are NOT medical advice, diagnosis, or treatment.</strong> Our platform provides general fitness and wellness information only. You must consult qualified healthcare professionals before beginning any exercise program, especially if you have pre-existing medical conditions, injuries, or health concerns.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Eligibility Requirements */}
              <section id="eligibility">
                <div className="flex items-start gap-4 mb-6">
                  <div className="w-8 h-8 bg-orange-500 text-white rounded-lg flex items-center justify-center font-bold text-sm">3</div>
                  <h2 className="text-3xl font-semibold text-slate-900">Eligibility Requirements</h2>
                </div>
                <div className="bg-slate-50 border border-slate-200 rounded-lg p-6 mb-6">
                  <h4 className="font-semibold text-slate-900 mb-3">You must meet ALL of the following requirements:</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <ul className="space-y-3 text-slate-700">
                      <li className="flex items-start gap-2">
                        <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></span>
                        <span>Be at least 18 years of age</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></span>
                        <span>Have legal capacity to enter binding contracts</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></span>
                        <span>Provide accurate and complete registration information</span>
                      </li>
                    </ul>
                    <ul className="space-y-3 text-slate-700">
                      <li className="flex items-start gap-2">
                        <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></span>
                        <span>Not be prohibited by applicable law from using our services</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></span>
                        <span>Have obtained medical clearance for physical exercise</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="w-2 h-2 bg-orange-500 rounded-full mt-2 flex-shrink-0"></span>
                        <span>Agree to comply with all applicable laws and regulations</span>
                      </li>
                    </ul>
                  </div>
                </div>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                  <p className="text-yellow-800 text-sm">
                    <strong>Note:</strong> Certain services may have additional age or health requirements. We reserve the right to verify eligibility and may request additional documentation.
                  </p>
                </div>
              </section>

              {/* User Responsibilities */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">3. User Responsibilities and Conduct</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  You agree to:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700 mb-4">
                  <li>Provide accurate and complete information during registration</li>
                  <li>Maintain the confidentiality of your account credentials</li>
                  <li>Use the platform only for lawful purposes</li>
                  <li>Respect the rights and privacy of other users</li>
                  <li>Follow all exercise instructions carefully and within your physical capabilities</li>
                  <li>Seek medical clearance before participating in any fitness program</li>
                </ul>
                <p className="text-slate-700 leading-relaxed">
                  You agree NOT to:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Share your account with others or create multiple accounts</li>
                  <li>Reproduce, distribute, or commercialize any content from the platform</li>
                  <li>Engage in any behavior that could harm other users or the platform</li>
                  <li>Use the platform if you have been medically advised against exercise</li>
                </ul>
              </section>

              {/* Health and Safety */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">4. Health and Safety Disclaimers</h2>
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-yellow-900">Exercise Risks</h4>
                      <p className="text-yellow-800 text-sm mt-1">
                        Physical exercise involves inherent risks including but not limited to muscle strains, cardiovascular stress, and potential injury. Participate at your own risk.
                      </p>
                    </div>
                  </div>
                </div>
                <p className="text-slate-700 leading-relaxed mb-4">
                  Before beginning any exercise program through REPZ, you must:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Obtain clearance from a qualified healthcare provider</li>
                  <li>Disclose any medical conditions, injuries, or physical limitations</li>
                  <li>Stop exercising immediately if you experience pain, dizziness, or discomfort</li>
                  <li>Understand that individual results may vary significantly</li>
                </ul>
              </section>

              {/* Payment Terms */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">5. Payment Terms and Subscriptions</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  REPZ offers various subscription tiers with different features and pricing. All fees are charged in advance and are non-refundable except as required by law.
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Subscription fees are billed automatically on a recurring basis</li>
                  <li>You may cancel your subscription at any time through your account settings</li>
                  <li>Cancellation takes effect at the end of your current billing period</li>
                  <li>Price changes will be communicated 30 days in advance</li>
                  <li>Failed payments may result in account suspension</li>
                </ul>
              </section>

              {/* Intellectual Property */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">6. Intellectual Property Rights</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  All content on REPZ, including but not limited to text, graphics, logos, workout programs, and software, is the property of REPZ or its licensors and is protected by intellectual property laws.
                </p>
                <p className="text-slate-700 leading-relaxed">
                  You are granted a limited, non-exclusive license to use the platform for personal, non-commercial purposes only. You may not reproduce, distribute, modify, or create derivative works from any platform content.
                </p>
              </section>

              {/* Limitation of Liability */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">7. Limitation of Liability</h2>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="h-5 w-5 text-red-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <h4 className="font-semibold text-red-900">Important Legal Notice</h4>
                      <p className="text-red-800 text-sm mt-1">
                        REPZ, its owners, employees, and affiliates shall not be liable for any direct, indirect, incidental, or consequential damages arising from your use of the platform or participation in any fitness programs.
                      </p>
                    </div>
                  </div>
                </div>
                <p className="text-slate-700 leading-relaxed">
                  This includes but is not limited to injuries, health complications, property damage, or any other losses. Your use of REPZ is entirely at your own risk, and you assume full responsibility for any consequences.
                </p>
              </section>

              {/* Termination */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">8. Account Termination</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  We reserve the right to terminate or suspend your account immediately, without prior notice, for any breach of these Terms of Service. Upon termination:
                </p>
                <ul className="list-disc pl-6 space-y-2 text-slate-700">
                  <li>Your right to use the platform ceases immediately</li>
                  <li>All data associated with your account may be deleted</li>
                  <li>No refunds will be provided for unused subscription time</li>
                  <li>You remain liable for any outstanding fees</li>
                </ul>
              </section>

              {/* Dispute Resolution */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">9. Dispute Resolution and Governing Law</h2>
                <p className="text-slate-700 leading-relaxed mb-4">
                  These terms shall be governed by and construed in accordance with the laws of [Your Jurisdiction]. Any disputes arising from these terms or your use of REPZ shall be resolved through binding arbitration.
                </p>
                <p className="text-slate-700 leading-relaxed">
                  You waive any right to participate in class action lawsuits or class-wide arbitration against REPZ.
                </p>
              </section>

              {/* Changes to Terms */}
              <section>
                <h2 className="text-2xl font-semibold text-slate-900 mb-4">10. Changes to Terms</h2>
                <p className="text-slate-700 leading-relaxed">
                  We reserve the right to modify these terms at any time. Material changes will be communicated via email or platform notification at least 30 days before taking effect. Continued use of REPZ after changes constitutes acceptance of the modified terms.
                </p>
              </section>

              {/* Contact Information */}
              <section className="bg-gradient-to-r from-slate-50 to-gray-50 border border-slate-200 rounded-xl p-8">
                <div className="text-center mb-6">
                  <div className="w-12 h-12 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl flex items-center justify-center mx-auto mb-4">
                    <Mail className="h-6 w-6 text-white" />
                  </div>
                  <h2 className="text-2xl font-semibold text-slate-900 mb-2">Questions About These Terms?</h2>
                  <p className="text-slate-600">We're here to help clarify any legal questions you may have.</p>
                </div>
                
                <div className="grid md:grid-cols-2 gap-6">
                  <div className="bg-white rounded-lg p-6 shadow-sm border border-slate-200">
                    <h4 className="font-semibold text-slate-900 mb-3">Legal Department</h4>
                    <div className="space-y-3 text-slate-700">
                      <div className="flex items-center gap-3">
                        <Mail className="h-4 w-4 text-orange-500" />
                        <span>legal@repz.com</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <Phone className="h-4 w-4 text-orange-500" />
                        <span>+1 (415) 992-9792 ext. 101</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-white rounded-lg p-6 shadow-sm border border-slate-200">
                    <h4 className="font-semibold text-slate-900 mb-3">Customer Support</h4>
                    <div className="space-y-3 text-slate-700">
                      <div className="flex items-center gap-3">
                        <Mail className="h-4 w-4 text-orange-500" />
                        <span>support@repz.com</span>
                      </div>
                      <p className="text-sm text-slate-600">Mon-Fri 6AM-8PM PST</p>
                    </div>
                  </div>
                </div>

                <div className="text-center mt-6 pt-6 border-t border-slate-200">
                  <p className="text-sm text-slate-500">
                    REPZ Coach Pro AI, LLC • San Francisco, CA • © 2025 All Rights Reserved
                  </p>
                </div>
              </section>

            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}